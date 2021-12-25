No setup aside from conda should be needed -- but setup commands are in notebook1 and notebook2, commented out.

# Conda Setup

conda create --name <YOUR_ENV_NAME> --file conda_env.txt

# Relevant Techniques

Saliency
Guided-GradCam
Deconvnet
Guided-BackProp

See `script_<technique>.py`


    'deconv': Deconvolution(vqa_resnet),
    'salien': Saliency(vqa_resnet),
    'gbackp': GuidedBackprop(vqa_resnet),
    'ggrdcm': GuidedGradCam(vqa_resnet, vqa_resnet.module.module.resnet_layer4.r_model.layer4),

$ python script_ULTIMATE.py deconv datainputindex.pickle 100 103 ans/new_deconvnet.pickle

python script_ULTIMATE.py ggrdcm datainputindex.pickle 100 150 ans-ultimate/ggrdcm-0100-0150.pickle \
&& python script_ULTIMATE.py gbackp datainputindex.pickle 100 150 ans-ultimate/gbackp-0100-0150.pickle \
&& python script_ULTIMATE.py salien datainputindex.pickle 100 150 ans-ultimate/salien-0100-0150.pickle \
&& python script_ULTIMATE.py deconv datainputindex.pickle 100 150 ans-ultimate/deconv-0100-0150.pickle; \ 
python script_ULTIMATE.py ggrdcm datainputindex.pickle 150 200 ans-ultimate/ggrdcm-0150-0200.pickle \
&& python script_ULTIMATE.py gbackp datainputindex.pickle 150 200 ans-ultimate/gbackp-0150-0200.pickle \
&& python script_ULTIMATE.py salien datainputindex.pickle 150 200 ans-ultimate/salien-0150-0200.pickle \
&& python script_ULTIMATE.py deconv datainputindex.pickle 150 200 ans-ultimate/deconv-0150-0200.pickle; \
python script_ULTIMATE.py ggrdcm datainputindex.pickle 200 250 ans-ultimate/ggrdcm-0200-0250.pickle \
&& python script_ULTIMATE.py gbackp datainputindex.pickle 200 250 ans-ultimate/gbackp-0200-0250.pickle \
&& python script_ULTIMATE.py salien datainputindex.pickle 200 250 ans-ultimate/salien-0200-0250.pickle \
&& python script_ULTIMATE.py deconv datainputindex.pickle 200 250 ans-ultimate/deconv-0200-0250.pickle; \
python script_ULTIMATE.py ggrdcm datainputindex.pickle 250 300 ans-ultimate/ggrdcm-0250-0300.pickle \
&& python script_ULTIMATE.py gbackp datainputindex.pickle 250 300 ans-ultimate/gbackp-0250-0300.pickle \
&& python script_ULTIMATE.py salien datainputindex.pickle 250 300 ans-ultimate/salien-0250-0300.pickle \
&& python script_ULTIMATE.py deconv datainputindex.pickle 250 300 ans-ultimate/deconv-0250-0300.pickle; \
python script_ULTIMATE.py ggrdcm datainputindex.pickle 300 350 ans-ultimate/ggrdcm-0300-0350.pickle \
&& python script_ULTIMATE.py gbackp datainputindex.pickle 300 350 ans-ultimate/gbackp-0300-0350.pickle \
&& python script_ULTIMATE.py salien datainputindex.pickle 300 350 ans-ultimate/salien-0300-0350.pickle \
&& python script_ULTIMATE.py deconv datainputindex.pickle 300 350 ans-ultimate/deconv-0300-0350.pickle; \
python script_ULTIMATE.py ggrdcm datainputindex.pickle 350 400 ans-ultimate/ggrdcm-0350-0400.pickle \
&& python script_ULTIMATE.py gbackp datainputindex.pickle 350 400 ans-ultimate/gbackp-0350-0400.pickle \
&& python script_ULTIMATE.py salien datainputindex.pickle 350 400 ans-ultimate/salien-0350-0400.pickle \
&& python script_ULTIMATE.py deconv datainputindex.pickle 350 400 ans-ultimate/deconv-0350-0400.pickle; \
python script_ULTIMATE.py ggrdcm datainputindex.pickle 400 450 ans-ultimate/ggrdcm-0400-0450.pickle \
&& python script_ULTIMATE.py gbackp datainputindex.pickle 400 450 ans-ultimate/gbackp-0400-0450.pickle \
&& python script_ULTIMATE.py salien datainputindex.pickle 400 450 ans-ultimate/salien-0400-0450.pickle \
&& python script_ULTIMATE.py deconv datainputindex.pickle 400 450 ans-ultimate/deconv-0400-0450.pickle; \
python script_ULTIMATE.py ggrdcm datainputindex.pickle 450 500 ans-ultimate/ggrdcm-0450-0500.pickle \
&& python script_ULTIMATE.py gbackp datainputindex.pickle 450 500 ans-ultimate/gbackp-0450-0500.pickle \
&& python script_ULTIMATE.py salien datainputindex.pickle 450 500 ans-ultimate/salien-0450-0500.pickle \
&& python script_ULTIMATE.py deconv datainputindex.pickle 450 500 ans-ultimate/deconv-0450-0500.pickle; \
python script_ULTIMATE.py ggrdcm datainputindex.pickle 500 600 ans-ultimate/ggrdcm-0500-0600.pickle \
&& python script_ULTIMATE.py gbackp datainputindex.pickle 500 600 ans-ultimate/gbackp-0500-0600.pickle \
&& python script_ULTIMATE.py salien datainputindex.pickle 500 600 ans-ultimate/salien-0500-0600.pickle \
&& python script_ULTIMATE.py deconv datainputindex.pickle 500 600 ans-ultimate/deconv-0500-0600.pickle; \
python script_ULTIMATE.py ggrdcm datainputindex.pickle 600 700 ans-ultimate/ggrdcm-0600-0700.pickle \
&& python script_ULTIMATE.py gbackp datainputindex.pickle 600 700 ans-ultimate/gbackp-0600-0700.pickle \
&& python script_ULTIMATE.py salien datainputindex.pickle 600 700 ans-ultimate/salien-0600-0700.pickle \
&& python script_ULTIMATE.py deconv datainputindex.pickle 600 700 ans-ultimate/deconv-0600-0700.pickle; \
python script_ULTIMATE.py ggrdcm datainputindex.pickle 700 800 ans-ultimate/ggrdcm-0700-0800.pickle \
&& python script_ULTIMATE.py gbackp datainputindex.pickle 700 800 ans-ultimate/gbackp-0700-0800.pickle \
&& python script_ULTIMATE.py salien datainputindex.pickle 700 800 ans-ultimate/salien-0700-0800.pickle \
&& python script_ULTIMATE.py deconv datainputindex.pickle 700 800 ans-ultimate/deconv-0700-0800.pickle; \
python script_ULTIMATE.py ggrdcm datainputindex.pickle 800 900 ans-ultimate/ggrdcm-0800-0900.pickle \
&& python script_ULTIMATE.py gbackp datainputindex.pickle 800 900 ans-ultimate/gbackp-0800-0900.pickle \
&& python script_ULTIMATE.py salien datainputindex.pickle 800 900 ans-ultimate/salien-0800-0900.pickle \
&& python script_ULTIMATE.py deconv datainputindex.pickle 800 900 ans-ultimate/deconv-0800-0900.pickle; \
python script_ULTIMATE.py ggrdcm datainputindex.pickle 900 1000 ans-ultimate/ggrdcm-0900-1000.pickle \
&& python script_ULTIMATE.py gbackp datainputindex.pickle 900 1000 ans-ultimate/gbackp-0900-1000.pickle \
&& python script_ULTIMATE.py salien datainputindex.pickle 900 1000 ans-ultimate/salien-0900-1000.pickle \
&& python script_ULTIMATE.py deconv datainputindex.pickle 900 1000 ans-ultimate/deconv-0900-1000.pickle; \