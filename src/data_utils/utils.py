import yaml
import os
import torch
import evaluate
# from data_utils.losses import SentEmb_loss

bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')

def load_model(model_name, classes=None):
    pass

def get_loss(config, weights = None, device = None):
    if config["type"] == 'CrossEntropyLoss':
        ignore_index = config.get("ignore_index", None)
        if config["weights"] != False:
            print("Using weights: ", weights)
            assert weights != None and device != None, "Weights and device must be provided"
            return torch.nn.CrossEntropyLoss(weight=weights).to(device)
        elif ignore_index != None:
            return torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        else: 
            return torch.nn.CrossEntropyLoss()
    elif config["type"] == 'SentEmb_loss':
        return SentEmb_loss
    

def get_scheduler(config, optimizer):
    name = config.get("type", None)

    if name == None: 
        print("Using no scheduler")
        return None

    print("Using scheduler: ", name)

    if name == 'ReduceOnPlateu':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    elif name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    elif name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['T_max'], eta_min=config['eta_min'])
    
    elif name == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=config['gamma'])
    
    elif name == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = config['T_0'], T_mult=config['T_m'], eta_min=config['eta_min'])
    
    else:
        raise Exception("Scheduler not found")

def get_optimer(config, model):
    optimizer_name = config["type"]
    lr = config["lr"]

    print("Using optimizer: ", optimizer_name)

    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name.lower() == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        raise Exception("Optimizer not found")

def createDir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def get_weights(weights_path):
    path2load = None
    for path in os.listdir(weights_path):
        if path[-4:] == ".pth":
            if path2load == None: 
                path2load = path
            elif int(path.split(".")[0]) > int(path2load.split(".")[0]):        
                path2load = path
    if path2load == None: 
        print("No weights found")
        return None

    print("Loading weights: ", path2load)
    return weights_path + path2load, int(path2load.split(".")[0])

def metrics_evaluation(pred, ref):
    metrics = {}
    try:
        metrics["bleu1"] = bleu.compute(predictions=pred, references=ref, max_order=1)["bleu"]
        metrics["bleu2"] = bleu.compute(predictions=pred, references=ref, max_order=2)["bleu"]
        metrics["rouge"] = rouge.compute(predictions=pred, references=ref)["rougeL"]
        metrics["meteor"] = meteor.compute(predictions=pred, references=ref)["meteor"]
    except ZeroDivisionError:
        metrics["bleu1"] = 0
        metrics["bleu2"] = 0
        metrics["rougeL"] = 0
        metrics["meteor"] = 0
    return metrics


import os.path as osp
def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def LoadConfig(test_name):
    with open("/fhome/gia07/Image_captioning/src/setups/" + test_name + ".yaml") as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    opt[test_name] = test_name
    opt["output_dir"] = "/fhome/gia07/Image_captioning/src/runs/" + test_name + "/images/"
    opt["train_output_dir"] = "/fhome/gia07/Image_captioning/src/runs/" + test_name + "/train_images/"
    opt["weights_dir"] = "/fhome/gia07/Image_captioning/src/runs/" + test_name + "/weights/"
    opt["root_dir"] = "/fhome/gia07/Image_captioning/src/runs/" + test_name + "/"

    createDir(opt["root_dir"])
    createDir(opt["weights_dir"])
    createDir(opt["output_dir"])
    createDir(opt["train_output_dir"])

    # aixo es per evitar que falli amb configs antics
    opt["network"]["params"]["teacher_forcing_ratio"] = opt["network"].get("params", {}).get("teacher_forcing_ratio", opt.get("teacher_forcing_ratio", 0))
    opt["network"]["params"]["rnn_layers"] = opt["network"].get("params", {}).get("rnn_layers",3)

    # opt["network"]["checkpoint"] = opt["network"].get("checkpoint", None)

    return opt


def LoadConfig_baseline(test_name):
    with open("/fhome/gia07/Image_captioning/src/setups-baseline/" + test_name + ".yaml") as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    opt[test_name] = test_name
    opt["root_dir"] = "/fhome/gia07/Image_captioning/src/runs-baseline/" + test_name + "/"
    opt["output_dir"] = opt["root_dir"] + "images/"
    opt["train_output_dir"] = opt["root_dir"] + "train_images/"
    opt["weights_dir"] = opt["root_dir"] + "weights/"

    createDir(opt["root_dir"])
    createDir(opt["weights_dir"])
    createDir(opt["output_dir"])
    createDir(opt["train_output_dir"])

    if opt["network"]["params"]["text_max_len"] == "default":
        opt["network"]["params"]["text_max_len"] = 201 if "char" in opt["datasets"]["type"] else 38

    opt["loss"] = opt.get("loss", {})
    opt["loss"]["type"] = opt["loss"].get("type", "CrossEntropyLoss")
    opt["loss"]["weights"] = opt["loss"].get("weights", "False")

    return opt


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

def patch_attention(m):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap


def upload_self_attention(save_output, input_words, idx2word):
    import wandb
    for i, (src, translation) in enumerate(zip(input_words, input_words[1:])):
        # print(save_output.outputs[i].shape)
        attn = save_output.outputs[i][0, 0, :, :]

        # print(translation)
        src = [idx2word[idx] for idx in src]
        translation = [idx2word[idx] for idx in translation[1:]]


        attn_data = []
        for m in range(attn.shape[0]):
            for n in range(attn.shape[1]):
                attn_data.append([n, m, src[n], translation[m], attn[m, n]])
        data_table = wandb.Table(data=attn_data, columns=["s_ind", "t_ind", "s_word", "t_word", "attn"])
        fields = {
            "s_index": "s_ind",
            "t_index": "t_ind",
            "sword": "s_word",
            "tword": "t_word",
            "attn": "attn"
        }
        wandb.log({
            f"my_nlp_viz_id_{i}": wandb.plot_table(
                            vega_spec_name="kylegoyette/nlp-attention-visualization",
                            data_table=data_table,
                            fields=fields
                            )
        })
    