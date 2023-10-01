import argparse
import torch
import AdaIN_net

if __name__ == '__main__':
    # Get arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-content_dir")
    parser.add_argument("-style_dir")
    parser.add_argument("-gamma")
    parser.add_argument("-e")
    parser.add_argument("-b")
    parser.add_argument("-l")
    parser.add_argument("-s")
    parser.add_argument("-p")
    parser.add_argument("-cuda")

    args = vars(parser.parse_args())

    epoch = int(args["e"])
    batch = int(args["b"])
    encoder_pth = args["l"]
    decoder_pth = args["s"]
    loss_plot = args["p"] #??????????????
    device = "cuda" if str(args["cuda"]).upper() == "Y" else "mps"  # change to cpu



    encoder_decoder = AdaIN_net.encoder_decoder()
    encoder_decoder.encoder.load_state_dict(torch.load(encoder_pth))

    model = AdaIN_net.AdaIN_net(encoder_decoder.encoder, encoder_decoder.decoder)

    model.train()
    model.to(device)


