import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, device="cuda"):
    return_info = ""

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                # summary[m_key]["output_shape"] = [
                #     [-1] + list(o.size())[1:] for o in output[0] if isinstance(o, torch.Tensor)
                # ]
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output if isinstance(o, torch.Tensor)
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    return_info += "----------------------------------------------------------------\n"

    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)","Output Shape", "Param #" )
    print(line_new)
    return_info += line_new +"\n"
    print("================================================================")
    return_info += "================================================================\n"
    temp_shape = list(x[0].shape)
    temp_shape[0] = batch_size
    line_new = "{:>20}  {:>25} {:>15}".format("input-0", str(temp_shape), "0")
    print(line_new)
    return_info += line_new +"\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        try:
            if type(summary[layer]["output_shape"][0]) is list:
                for item in summary[layer]["output_shape"]:
                    total_output += np.prod(item)
            else:
                total_output += np.prod(summary[layer]["output_shape"])
        except:
            print("here is a bug")
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)
        return_info += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    return_info += "================================================================\n"
    return_info += "Total params: {0:,}\n".format(total_params)
    return_info += "Trainable params: {0:,}\n".format(trainable_params)
    return_info += "Non-trainable params: {0:,}\n".format(total_params - trainable_params)
    return_info += "----------------------------------------------------------------\n"
    return_info += "Input size (MB): %0.2f\n" % total_input_size
    return_info += "Forward/backward pass size (MB): %0.2f\n" % total_output_size
    return_info += "Params size (MB): %0.2f\n" % total_params_size
    return_info += "Estimated Total Size (MB): %0.2f\n" % total_size
    return_info += "----------------------------------------------------------------\n"
    return_info += "================================================================\n"
    return return_info
    # return summary
