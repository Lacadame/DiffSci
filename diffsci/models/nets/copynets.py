from diffsci.models.nets.autoencoderldm3d import AutoencoderKL
from diffsci.models.nets.vaenet import VAENet


def copy_autoencoderkl_to_vaenet(
    autoencoder_kl: AutoencoderKL,
    vaenet: VAENet
):
    """
    Copy parameters from an AutoencoderKL model to a VAENet model.
    
    Args:
        autoencoder_kl: Source AutoencoderKL model
        vaenet: Target VAENet model
    """
    # Copy decoder parameters
    for key, param in autoencoder_kl.decoder.named_parameters():
        if key in dict(vaenet.decoder.named_parameters()):
            # Get the corresponding parameter in vaenet
            vaenet_param = dict(vaenet.decoder.named_parameters())[key]
            if vaenet_param.shape == param.shape:
                # Copy the parameter values
                vaenet_param.data.copy_(param.data)
                print(f"Successfully copied decoder.{key}")
            else:
                print(f"Shape mismatch for decoder.{key}: {vaenet_param.shape} vs {param.shape}")
        else:
            wrapped_key = key.replace('.weight', '.conv.weight').replace('.bias', '.conv.bias')
            if wrapped_key in dict(vaenet.decoder.named_parameters()):
                vaenet_wrapped_param = dict(vaenet.decoder.named_parameters())[wrapped_key]
                if vaenet_wrapped_param.shape == param.shape:
                    # Copy the parameter values to the wrapped parameter
                    vaenet_wrapped_param.data.copy_(param.data)
                    print(f"Successfully copied decoder.{key} to wrapped parameter decoder.{wrapped_key}")
                else:
                    print(f"Shape mismatch for decoder.{key}: {vaenet_wrapped_param.shape} vs {param.shape}")
            else:
                print(f"Parameter decoder.{key} not found in vaenet.decoder")
    
    # Copy post_quant_conv parameters
    try:
        vaenet.decoder.post_quant_conv.weight.data.copy_(autoencoder_kl.post_quant_conv.weight.data)
        vaenet.decoder.post_quant_conv.bias.data.copy_(autoencoder_kl.post_quant_conv.bias.data)
        print("Successfully copied post_quant_conv parameters")
    except Exception as e:
        print(f"Error copying post_quant_conv parameters: {e}")
    
    # Copy encoder parameters
    for key, param in autoencoder_kl.encoder.named_parameters():
        if key in dict(vaenet.encoder.named_parameters()):
            # Get the corresponding parameter in vaenet
            vaenet_param = dict(vaenet.encoder.named_parameters())[key]
            if vaenet_param.shape == param.shape:
                # Copy the parameter values
                vaenet_param.data.copy_(param.data)
                print(f"Successfully copied encoder.{key}")
            else:
                print(f"Shape mismatch for encoder.{key}: {vaenet_param.shape} vs {param.shape}")
        else:
            wrapped_key = key.replace('.weight', '.conv.weight').replace('.bias', '.conv.bias')
            if wrapped_key in dict(vaenet.encoder.named_parameters()):
                vaenet_wrapped_param = dict(vaenet.encoder.named_parameters())[wrapped_key]
                if vaenet_wrapped_param.shape == param.shape:
                    # Copy the parameter values to the wrapped parameter
                    vaenet_wrapped_param.data.copy_(param.data)
                    print(f"Successfully copied encoder.{key} to wrapped parameter encoder.{wrapped_key}")
                else:
                    print(f"Shape mismatch for encoder.{key}: {vaenet_wrapped_param.shape} vs {param.shape}")
            else:
                print(f"Parameter encoder.{key} not found in vaenet.encoder")
    
    # Copy quant_conv parameters
    try:
        vaenet.encoder.quant_conv.weight.data.copy_(autoencoder_kl.quant_conv.weight.data)
        vaenet.encoder.quant_conv.bias.data.copy_(autoencoder_kl.quant_conv.bias.data)
        print("Successfully copied quant_conv parameters")
    except Exception as e:
        print(f"Error copying quant_conv parameters: {e}")
