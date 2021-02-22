import argparse
from model import load_model, save_model, create_model, train_model, test_model
from data import create_data, format_data

DEFAULT_DATASET = "OOK"
DEFAULT_COMPONENT = "LPF"
DEFAULT_TYPE = "CNN"
DEFAULT_BATCH_SIZE = 128
DEFAULT_TIMESTEPS = 64
DEFAULT_EPOCHS = (500, 250)

def parse_args():
    parser = argparse.ArgumentParser(description="""
    This script is used to build and test CNN and LSTM models for SDR component resiliency. 
    """)
    parser.add_argument("--load", "-l", help="Name of model to load. Leave empty to build new model. Will override --save argument")
    # parser.add_argument("--save", "-s", help="Name of model to save")
    parser.add_argument("--train", "-t", action="store_true", help="Determines whether or not to train the model")
    parser.add_argument("--freq", "-f", action="store_true", help="Determines whether or not to do a frequency response of the model. Will override --train arg")
    parser.add_argument("--plot", "-p", action="store_true", help="Determines whether or not to plot the results after testing the model")
    parser.add_argument("--dataset", "-d", default=DEFAULT_DATASET, help="Dataset used to train/test model. Acceptable inputs: OOK, Audio")
    parser.add_argument("--component", "-c", default=DEFAULT_COMPONENT, help="Component used to train/test model. Acceptable inputs: LPF, Mixer, DDC")
    parser.add_argument("--type", "-ty", default=DEFAULT_TYPE, help="Sets the model architecture. Acceptable inputs: CNN, LSTM")
    parser.add_argument("--batch-size", "-b", default=DEFAULT_BATCH_SIZE, help="Sets the batch size for the network. Accepts a positve integer")
    parser.add_argument("--timesteps", "-ts", default=DEFAULT_TIMESTEPS, help="Component used to train/test model. Accepts a positve integer")
    parser.add_argument("--epochs", "-e", default=DEFAULT_EPOCHS, help="Sets the number of epochs to train. Accepts a tuple of 2 positve integers for Adadelta and Adam epochs respectively")
    parser.add_argument("--dense-nodes", "-dn", default=50, help="Sets the number of dense layer nodes. Accepts a positive integer")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    model = None
    model_name = None
    if args.load:
        model = load_model(args.load)
        if not model:
            exit()
        
        model_name = args.load

        if args.timesteps == DEFAULT_TIMESTEPS:
            args.timesteps = model.layers[0].input_shape[1]

    history = None

    component_name = None
    if args.component == "LPF":
        component_name = "Low Pass Filter"
    elif args.component == "Mixer":
        component_name = "Mixer"
    elif args.component == "DDC":
        component_name = "DDC"

    
    if not model or args.train:
        data_obj_train = create_data(args.dataset, args.component)
        (x_train, y_train) = format_data(data_obj_train, args.timesteps)

        if not model:
            model = create_model(args.type, x_train.shape, args.dense_nodes)
            model_name = save_model(model)

        if args.epochs[0] > 0:
            (model, history) = train_model(model, x_train, y_train, args.epochs[0], model_name, 'adadelta', args.batch_size)

            data_obj_test = create_data(args.dataset, args.component)
            (x_test, y_test) = format_data(data_obj_test, args.timesteps)
            test_model(model, x_test, y_test, data_obj_test, history, args.batch_size, 'Adadelta', component_name)

        if args.epochs[1] > 0:
            (model, history) = train_model(model, x_train, y_train, args.epochs[1], 'adam', args.batch_size)

        save_model(model, model_name)
    
    data_obj_test = create_data(args.dataset, args.component)
    (x_test, y_test) = format_data(data_obj_test, args.timesteps)

    test_model(model, x_test, y_test, data_obj_test, history, args.batch_size, model_name,'Adam', component_name)




