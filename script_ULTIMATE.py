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

##############################################
import sys
attr_method = sys.argv[1]
attr_all = {
    'deconv': Deconvolution(vqa_resnet),
    'salien': Saliency(vqa_resnet),
    'gbackp': GuidedBackprop(vqa_resnet),
    'ggrdcm': GuidedGradCam(vqa_resnet, vqa_resnet.module.module.resnet_layer4.r_model.layer4),
}
assert attr_method in attr_all

if USE_INTEPRETABLE_EMBEDDING_LAYER:
    attr = attr_all[attr_method]

else:
    attr = LayerIntegratedGradients(vqa_resnet, [vqa_resnet.module.input_maps["v"], vqa_resnet.module.module.text.embedding])


default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#252b36'),
                                                  (1, '#000000')], N=256)

def vqa_interpret_once(image_features, question_parsed, target_index):
    q, q_len = encode_question(question_parsed)
    
    inputs = (q.unsqueeze(0), q_len.unsqueeze(0))
    if USE_INTEPRETABLE_EMBEDDING_LAYER:
        q_input_embedding = interpretable_embedding.indices_to_embeddings(q).unsqueeze(0)
        inputs = (image_features, q_input_embedding)
    else:            
        inputs = (image_features, q.unsqueeze(0))
        
    ans = vqa_resnet(*inputs, q_len.unsqueeze(0))
        
    # Make a prediction. The output of this prediction will be visualized later.
    pred, answer_idx = F.softmax(ans, dim=1).data.cpu().max(dim=1)

    attributions = attr.attribute(inputs=inputs, ###?
                                target=target_index,
                                additional_forward_args=q_len.unsqueeze(0))

    attributions_img = np.transpose(attributions[0].squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    
    return {
        'model_pred': answer_words[ answer_idx ],
        'model_conf': pred[0].item(),

        'attr_target': answer_words[ target_index ],
        'attr_heatmap': attributions_img
    }

def vqa_resnet_interpret(image_filename, questions, answerkeys):
    """
     image_filename is filepath, must be verified
     questions are list of raw questions
     answerkeys are the list of answers that are valid

     Returns [
         {
            'input_imgname': filename of image,
            'input_question': a string, a question -- unmodified from input,
            'input_answerkey': correct answer
            'model_pred': natural language answer of the model
            'model_conf': confidence of the model, softmax
            'attr_target': attribution target class
            'attr_heatmap': heatmap of 448x448x3 numpy array
         }
     ]
    """
    attrs = []
    img = Image.open(image_filename).convert('RGB')    
    image_features = image_to_features(img).requires_grad_().to(device)

    for question, answerkey in zip(questions, answerkeys):
        question_parsed = question.lower().replace('?', '')
        target_index = answer_to_index[answerkey]
        attrs.append(vqa_interpret_once(image_features, question_parsed, target_index))
        attrs[-1]['input_imgname'] = image_filename
        attrs[-1]['input_question'] = question
        attrs[-1]['input_answerkey'] = answerkey

        if attrs[-1]['model_pred'] != answerkey:
            target_index = answer_to_index[attrs[-1]['model_pred']]
            attrs.append(vqa_interpret_once(image_features, question_parsed, target_index))
            attrs[-1]['input_imgname'] = image_filename
            attrs[-1]['input_question'] = question
            attrs[-1]['input_answerkey'] = answerkey

    return attrs

import pickle
import os
def check_and_clean_inputs(filename, start, stop):
    """
    [ (verified filepath, [raw questions], [verified answers]) ]
    """
    with open(filename, 'rb') as handle:
        inputs = pickle.load(handle)

    to_return = []
    start = max(0, min(start, len(inputs)-1))
    stop = max(1, min(stop, len(inputs)))

    for i in range(start, stop):
        verified_filepath = inputs[i][0]
        assert os.path.isfile(verified_filepath)

        raw_question_list = []
        verifed_answers_list = []

        assert len(inputs[i][1]) == len(inputs[i][2])
        for j in range(len(inputs[i][2])):
            if inputs[i][2][j] in answer_to_index:
                raw_question_list.append(inputs[i][1][j])
                verifed_answers_list.append(inputs[i][2][j])

        if len(raw_question_list) > 0:
            to_return.append((verified_filepath, raw_question_list, verifed_answers_list))
    
    return to_return
    
################################

import sys
filename_input = sys.argv[2]
start = int(sys.argv[3])
stop = int(sys.argv[4])
filename_output = sys.argv[5]

inputs = check_and_clean_inputs(filename_input, start, stop)
ans = []

for i in range(len(inputs)):
    input = inputs[i]
    print('progress', i, len(inputs))
    ans.extend(vqa_resnet_interpret(input[0], input[1], input[2]))

with open(filename_output, 'wb') as handle:
    pickle.dump(ans, handle)

################################

if USE_INTEPRETABLE_EMBEDDING_LAYER:
    remove_interpretable_embedding_layer(vqa_resnet, interpretable_embedding)