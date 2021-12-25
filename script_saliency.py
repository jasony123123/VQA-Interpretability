import warnings
warnings.filterwarnings("ignore")

import os, sys

sys.path.append("pytorch-vqa/")
sys.path.append("pytorch-resnet/")
import threading
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import resnet  # from pytorch-resnet

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

from model import Net, apply_attention, tile_2d_over_nd # from pytorch-vqa
from utils import get_transform # from pytorch-vqa

from captum.attr import (
    IntegratedGradients,
    LayerIntegratedGradients,
    Deconvolution,
    LayerGradCam,
    GuidedGradCam,
    GuidedBackprop,
    Saliency,
    TokenReferenceBase,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
    visualization
)
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


saved_state = torch.load('2017-08-04_00.55.19.pth', map_location=device)
# reading vocabulary from saved model
vocab = saved_state['vocab']

# reading word tokens from saved model
token_to_index = vocab['question']

# reading answers from saved model
answer_to_index = vocab['answer']

num_tokens = len(token_to_index) + 1

# reading answer classes from the vocabulary
answer_words = ['unk'] * len(answer_to_index)
for w, idx in answer_to_index.items():
    answer_words[idx]=w


vqa_net = torch.nn.DataParallel(Net(num_tokens))
vqa_net.load_state_dict(saved_state['weights'])
vqa_net.to(device)
vqa_net.eval()


def encode_question(question):
    """ Turn a question into a vector of indices and a question length. Unrecognized turned into 0."""
    question_arr = question.lower().split()
    vec = torch.zeros(len(question_arr), device=device).long()
    for i, token in enumerate(question_arr):
        index = token_to_index.get(token, 0)
        vec[i] = index
    return vec, torch.tensor(len(question_arr), device=device)


class ResNetLayer4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.r_model = resnet.resnet152(pretrained=True) # [18, 34, 50, 101, 150]
        self.r_model.eval()
        self.r_model.to(device)

        self.buffer = {}
        lock = threading.Lock()

        # Since we only use the output of the 4th layer from the resnet model and do not
        # need to do forward pass all the way to the final layer we can terminate forward
        # execution in the forward hook of that layer after obtaining the output of it.
        # For that reason, we can define a custom Exception class that will be used for
        # raising early termination error.
        def save_output(module, input, output):
            with lock:
                self.buffer[output.device] = output

        self.r_model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.r_model(x)          
        return self.buffer[x.device]

class VQA_Resnet_Model(Net):
    def __init__(self, embedding_tokens):
        super().__init__(embedding_tokens)
        self.resnet_layer4 = ResNetLayer4()
    
    def forward(self, v, q, q_len):
        q = self.text(q, list(q_len.data))
        v = self.resnet_layer4(v)

        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        a = self.attention(v, q)
        v = apply_attention(v, a)

        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)
        return answer


USE_INTEPRETABLE_EMBEDDING_LAYER = True  # set to True for option (1)


vqa_resnet = VQA_Resnet_Model(vqa_net.module.text.embedding.num_embeddings)

# wrap the inputs into layers incase we wish to use a layer method
vqa_resnet = ModelInputWrapper(vqa_resnet)

# `device_ids` contains a list of GPU ids which are used for paralelization supported by `DataParallel`
vqa_resnet = torch.nn.DataParallel(vqa_resnet)

# saved vqa model's parameters
partial_dict = vqa_net.state_dict()

state = vqa_resnet.module.state_dict()
state.update(partial_dict)
vqa_resnet.module.load_state_dict(state)

vqa_resnet.to(device)
vqa_resnet.eval()

# This is original VQA model without resnet. Removing it, since we do not need it
del vqa_net


if USE_INTEPRETABLE_EMBEDDING_LAYER:
    interpretable_embedding = configure_interpretable_embedding_layer(vqa_resnet, 'module.module.text.embedding')


image_size = 448  # scale image to given size and center
central_fraction = 1.0

transform = get_transform(image_size, central_fraction=central_fraction)
    
def image_to_features(img):
    img_transformed = transform(img)
    img_batch = img_transformed.unsqueeze(0).to(device)
    return img_batch


PAD_IND = token_to_index['pad']
token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)


# this is necessary for the backpropagation of RNNs models in eval mode
torch.backends.cudnn.enabled=False


if USE_INTEPRETABLE_EMBEDDING_LAYER:
    attr = Saliency(vqa_resnet)
else:
    attr = LayerIntegratedGradients(vqa_resnet, [vqa_resnet.module.input_maps["v"], vqa_resnet.module.module.text.embedding])


default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#252b36'),
                                                  (1, '#000000')], N=256)


def vqa_resnet_interpret(image_filename, questions, targets):
    """
        INPUT:
            image_filename: filepath to the image
            questions: list of strings, each is a question
            targets: list of target answers, one per question, for the attribution to analyze
        OUTPUT:
            List of Maps
                [
                    {
                        'input_imgname': filename of image,
                        'input_question': a string, aquestion,
                        'model_prediction_words': answer predicted by model
                        'model_prediction_confidence': softmax score of the model for the answer
                        'attr_target': attribution target answer,
                        'attr_heatmap': attribution heatmap,
                    },
                    ...
                ]
    """
    returned_attributions = []
    img = Image.open(image_filename).convert('RGB')
    original_image = transforms.Compose([transforms.Scale(int(image_size / central_fraction)),
                                   transforms.CenterCrop(image_size), transforms.ToTensor()])(img) 
    
    image_features = image_to_features(img).requires_grad_().to(device)
    for question, target in zip(questions, targets):
        q, q_len = encode_question(question)
        
        # generate reference for each sample
        q_reference_indices = token_reference.generate_reference(q_len.item(), device=device).unsqueeze(0)

        inputs = (q.unsqueeze(0), q_len.unsqueeze(0))
        if USE_INTEPRETABLE_EMBEDDING_LAYER:
            q_input_embedding = interpretable_embedding.indices_to_embeddings(q).unsqueeze(0)
            q_reference_baseline = interpretable_embedding.indices_to_embeddings(q_reference_indices).to(device)

            inputs = (image_features, q_input_embedding)
            baselines = (image_features * 0.0, q_reference_baseline)
            
        else:            
            inputs = (image_features, q.unsqueeze(0))
            baselines = (image_features * 0.0, q_reference_indices)
            
        ans = vqa_resnet(*inputs, q_len.unsqueeze(0))
            
        # Make a prediction. The output of this prediction will be visualized later.
        pred, answer_idx = F.softmax(ans, dim=1).data.cpu().max(dim=1)
        if target == None or target not in answer_to_index:
            target_idx = answer_idx
        else:
            target_idx = answer_to_index[target]


        ###!
        attributions = attr.attribute(inputs=inputs,
                                    # baselines=baselines,
                                    target=target_idx,
                                    additional_forward_args=q_len.unsqueeze(0),
                                    # n_steps=1
                                    )
        # # Visualize text attributions
        # text_attributions_norm = attributions[1].sum(dim=2).squeeze(0).norm()
        # vis_data_records = [visualization.VisualizationDataRecord(
        #                         attributions[1].sum(dim=2).squeeze(0) / text_attributions_norm,
        #                         pred[0].item(),
        #                         answer_words[ answer_idx ],
        #                         answer_words[ answer_idx ],
        #                         target,
        #                         attributions[1].sum(),       
        #                         question.split(),
        #                         0.0)]
        # visualization.visualize_text(vis_data_records)

        # # visualize image attributions
        # original_im_mat = np.transpose(original_image.cpu().detach().numpy(), (1, 2, 0))
        attributions_img = np.transpose(attributions[0].squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        
        # visualization.visualize_image_attr_multiple(attributions_img, original_im_mat, 
        #                                             ["original_image", "heat_map"], ["all", "absolute_value"], 
        #                                             titles=["Original Image", "Attribution Magnitude"],
        #                                             cmap=default_cmap,
        #                                             show_colorbar=True)
        # print('Text Contributions: ', attributions[1].sum().item())
        # print('Image Contributions: ', attributions[0].sum().item())
        # print('Total Contribution: ', attributions[0].sum().item() + attributions[1].sum().item())
        returned_attributions.append({
            'input_imgname': image_filename,
            'input_question': question,
            'model_prediction_words': answer_words[ answer_idx ],
            'model_prediction_confidence': pred[0].item(),
            'attr_target': answer_words[ target_idx ],
            'attr_heatmap': attributions_img,
        })
    return returned_attributions

def vqa_resnet_interpret_batch(inputs):
    """
        INPUT
            List of tuples.
            [
                (
                    image_filename
                    list_of_questions
                    list_of_targets
                )
            ]
        RETURNS
        [
            [ # all for the same image
                {
                    input_img_name,
                    question
                    model outputs
                    attribution outputs
                },
                ...
            ]
            [
                # another image
            ]
        ]
    """
    ret = []
    for i in range(len(inputs)):
        print('batch progress', i, len(inputs))
        input = inputs[i]
        attr = vqa_resnet_interpret(input[0], input[1], input[2])
        ret.append(attr)
    return ret

################################
# Start Editing Here           #
################################

import pickle

with open('model_data/batched_index_list_mscoco_val2014_MultipleChoice.pickle', 'rb') as handle:
    inputs = pickle.load(handle)

inputs = inputs[:10]

ans = vqa_resnet_interpret_batch(inputs)

with open('ans_saliency.pickle', 'wb') as handle:
    pickle.dump(ans, handle, protocol=pickle.HIGHEST_PROTOCOL)

################################
# Stop Editing Here            #
################################

if USE_INTEPRETABLE_EMBEDDING_LAYER:
    remove_interpretable_embedding_layer(vqa_resnet, interpretable_embedding)