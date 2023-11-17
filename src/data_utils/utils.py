import yaml
import os
import torch
import evaluate

bleu = evaluate.load('bleu')
meteor = evaluate.load('meteor')
rouge = evaluate.load('rouge')

def load_model(model_name, classes=None):
    pass

def get_optimer(optimizer_name, model, lr):
    if optimizer_name == 'ADAM':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'ADAGRAD':
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    else:
        raise Exception("Optimizer not found")

def createDir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def LoadConfig(test_name):
    with open("/fhome/gia07/Image_captioning/src/setups/" + test_name + ".yaml") as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    opt[test_name] = test_name
    opt["output_dir"] = "/fhome/gia07/Image_captioning/src/runs/" + test_name + "/images/"
    opt["weights_dir"] = "/fhome/gia07/Image_captioning/src/runs/" + test_name + "/weights/"
    opt["root_dir"] = "/fhome/gia07/Image_captioning/src/runs/" + test_name + "/"

    createDir(opt["root_dir"])
    createDir(opt["weights_dir"])
    createDir(opt["output_dir"])

    # opt["network"]["checkpoint"] = opt["network"].get("checkpoint", None)

    return opt

def get_weights(weights_path):
    path2load = None
    for path in os.listdir(weights_path):
        if path[-4:] == ".pth":
            if path2load == None: 
                path2load = path
            elif int(path.split("epoch_")[1].split(".")[0]) > int(path2load.split("epoch_")[1].split(".")[0]):        
                path2load = path
    if path2load == None: 
        print("No weights found")
        return None

    return weights_path + path2load

# def metrics_evaluation(pred, ref):
#     bleu1 = bleu.compute(predictions=pred, references=ref, max_order=1)
#     bleu2 = bleu.compute(predictions=pred, references=ref, max_order=2)
#     res_r = rouge.compute(predictions=pred, references=ref)
#     res_m = meteor.compute(predictions=pred, references=ref)

#     print(f"BLEU-1:{bleu1['bleu']*100:.1f}%, BLEU2:{bleu2['bleu']*100:.1f}%, ROUGE-L:{res_r['rougeL']*100:.1f}%, METEOR:{res_m['meteor']*100:.1f}%")

def metrics_evaluation(pred, ref):
    metrics = {}
    metrics["bleu1"] = bleu.compute(predictions=pred, references=ref, max_order=1)["bleu"]
    metrics["bleu2"] = bleu.compute(predictions=pred, references=ref, max_order=2)["bleu"]
    metrics["rougeL"] = rouge.compute(predictions=pred, references=ref)["rougeL"]
    metrics["meteor"] = meteor.compute(predictions=pred, references=ref)["meteor"]
    return metrics
    