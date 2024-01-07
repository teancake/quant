import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)



import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import quant_data_util
from lstm_model import get_model, device
import matplotlib.pyplot as plt

from utils.log_util import get_logger
from utils.stock_zh_a_util import is_trade_date

logger = get_logger(__name__)

def test_dataloader(dataloader, data_y):
    print("testing dataloader")
    for epoch in range(10):
        result = []
        for i, data in enumerate(dataloader):
            input, label = data
            if i == 23:
                print("i {}, label {}".format(i, label[0][0]))
            result.extend(label)
        print("result len {}".format(len(result)))
        print("53rd element of result is {}".format(result[53]))
    print("53rd element of data y is {}".format(data_y[53]))
    print("testing dataloader finished. ")



def train(args):
    sequence_length = 10
    feature_top_n = 20

    batch_size = args.batch_size
    learning_rate = args.lr
    num_layers = args.num_layers
    dropout = args.dropout
    hidden_size = args.hidden_size
    epoch_num = args.epoch_num
    use_roc_label = args.use_roc_label
    model_name = args.model_name
    label_name = args.label_name


    model_config = {
        "model_name": model_name,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout
    }

    train_file_name, pred_file_name = quant_data_util.get_train_pred_file_names(args.ds)
    sequential_data_file_name = quant_data_util.generate_sequential_data_file_name(args.ds)

    sequential_data = quant_data_util.get_sequential_data(train_file_name, pred_file_name,
                                                          sequential_data_file_name=sequential_data_file_name,
                                                          sequence_length=sequence_length,
                                                          use_roc_label=use_roc_label, feature_top_n=feature_top_n,
                                                          label_name=label_name)

    train_data_seq, train_data_cat, train_data_y, train_data_ext, test_data_seq, test_data_cat, test_data_y, test_data_ext, pred_data_seq, pred_data_cat, pred_data_ext = sequential_data

    print("data loaded")
    print("train data seq shape {},  cat shape {}, train y shape {}, train y mean {}, variance {}".format(train_data_seq.shape,  train_data_cat.shape, train_data_y.shape, np.mean(train_data_y), np.var(train_data_y)))
    print("test data seq shape {}, cat shape {}, test y shape {}, test y mean {}, variance {}".format(test_data_seq.shape, test_data_cat.shape, test_data_y.shape, np.mean(test_data_y), np.var(test_data_y)))
    print("pred data seq shape {}, cat shape {}, append x shape {}".format(pred_data_seq.shape, pred_data_cat.shape, pred_data_ext.shape))
    print("pred data ext {}".format(pred_data_ext))

    train_data_seq = torch.tensor(train_data_seq, dtype=torch.float32).to(device)
    train_data_cat = torch.tensor(train_data_cat, dtype=torch.int).to(device)
    train_data_y = torch.tensor(train_data_y, dtype=torch.float32).to(device)
    test_data_seq = torch.tensor(test_data_seq, dtype=torch.float32).to(device)
    test_data_cat = torch.tensor(test_data_cat, dtype=torch.int).to(device)
    test_data_y = torch.tensor(test_data_y, dtype=torch.float32).to(device)
    pred_data_seq = torch.tensor(pred_data_seq, dtype=torch.float32).to(device)
    pred_data_cat = torch.tensor(pred_data_cat, dtype=torch.int).to(device)
    train_dataset = TensorDataset(train_data_seq, train_data_cat, train_data_y)
    val_dataset = TensorDataset(test_data_seq, test_data_cat, test_data_y)
    pred_dataset = TensorDataset(pred_data_seq, pred_data_cat, torch.zeros(pred_data_seq.shape[0], pred_data_seq.shape[1], 1))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    # note: turn off shuffle, and disable drop last for val_loader and pred_loader,
    # because the order and length should be preserved
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)


    seq_input_size = train_data_seq.shape[-1]
    cat_input_size = train_data_cat.shape[-1]

    output_size = train_data_y.shape[-1]

    model_config["input_size"] = seq_input_size
    model_config["output_size"] = output_size
    model_config["cat_input_size"] = cat_input_size
    model = get_model(model_config)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("model created {}".format(model))

    # Run the training loop
    metric = []
    val_outputs = []
    for epoch in range(0, epoch_num):
        print("Starting epoch {}".format(epoch + 1))

        train_losses = []
        model.train(True)
        for batch_num, data in enumerate(train_loader):

            seq_inputs, cat_inputs, targets = data
            seq_inputs = seq_inputs.to(device)
            cat_inputs = cat_inputs.to(device)
            targets = targets.to(device)
            # print("input shape {}, target shape {}, h0 shape {}, h1 shape {}".format(inputs.shape, targets.shape, h[0].shape, h[1].shape))
            loss = model.get_loss(targets=targets, seq_inputs=seq_inputs, cat_inputs=cat_inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if batch_num + 1 % 50 == 0:
                print("Loss after batch {}: {}".format(batch_num + 1, np.mean(train_losses)))
        print("Training process in epoch {} has finished. Evaluation started.".format(epoch + 1))

        val_losses = []
        val_labels = []
        val_outputs = []
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                seq_inputs, cat_inputs, targets = data
                seq_inputs = seq_inputs.to(device)
                cat_inputs = cat_inputs.to(device)
                targets = targets.to(device)
                outputs = model.get_outputs(seq_inputs=seq_inputs, cat_inputs=cat_inputs)
                # print("voutputs shape {}, vlabels shape {}".format(voutputs.shape, vlabels.shape))
                # only the last day matters
                # vloss = loss_fn(voutputs[:,-1], vlabels[:,-1])
                # all predictions matters
                vloss = model.get_loss(targets=targets, seq_inputs=seq_inputs, cat_inputs=cat_inputs)
                val_labels.extend(targets[:, -1].squeeze().detach().cpu().numpy())
                val_outputs.extend(outputs[:, -1].squeeze().detach().cpu().numpy())
                # print("vloss {}, mse {}".format(vloss, mean_squared_error(val_outputs, val_labels)))
                val_losses.append(vloss.item())
        print("epoch {} train loss {}, val loss {}".format(epoch + 1, np.mean(train_losses), np.mean(val_losses)))
        print("val_labels {} ... {}".format(val_labels[0:10], val_labels[-10:]))
        print("val_outputs {} ... {}".format(val_outputs[0:10], val_outputs[-10:]))
        metric.append([epoch + 1, np.mean(train_losses), np.mean(val_losses)])


    print("metrics {}".format(metric))
    metric = np.array(metric)
    run_id = quant_data_util.get_run_id(prefix=args.model_name)
    plt.plot(metric[:, 0], metric[:, 1], label="train loss")
    plt.plot(metric[:, 0], metric[:, 2], linestyle='--', label="val loss")
    plt.legend(loc="upper right")
    plt.savefig('metric_{}.png'.format(run_id))


    label_name_pred = "{}_pred".format(label_name)
    test_data_ext[label_name_pred] = np.array(val_outputs)
    quant_data_util.compute_precision_recall_updated(test_data_ext[label_name].values.astype(float), test_data_ext[label_name_pred])
    #
    # quant_data_util.fill_ext_with_predictions(test_data_ext, pred, use_roc_label)
    # quant_data_util.compute_precision_recall(test_data_ext)
    pd.set_option('display.max_columns', 20)
    print(test_data_ext.head(10))
    # print(test_data_y[0:10])

    ds = args.ds
    save_prediction_result = args.save_prediction_result
    if save_prediction_result:
        logger.info("saving validation results")
        quant_data_util.save_prediction_to_db(test_data_ext, ds=ds, model_name=model_name, stage="validation",
                                              pred_name=label_name, pred_val_col_name=label_name_pred,
                                              label_col_name=label_name,
                                              run_id=run_id)

    print("now make predictions")
    pred_outputs = []
    model.eval()
    with torch.no_grad():
        for batch_num, data in enumerate(pred_loader):
            seq_inputs, cat_inputs, _ = data
            seq_inputs = seq_inputs.to(device)
            cat_inputs = cat_inputs.to(device)
            outputs = model.get_outputs(seq_inputs=seq_inputs, cat_inputs=cat_inputs)
            pred_outputs.extend(outputs[:, -1].squeeze().detach().cpu().numpy())

    # quant_data_util.fill_ext_with_predictions(pred_data_ext, pred, use_roc_label)
    pred_data_ext[label_name_pred] = pred_outputs
    print("now only use the latest date and find the ups")
    pred_data_ext = pred_data_ext[pred_data_ext["日期"] == pred_data_ext["日期"].max()].copy()
    pred_data_ext.sort_values(by=[label_name_pred], inplace=True, ascending=False)

    print(pred_data_ext.head(20).to_string())

    if args.save_prediction_result:
        quant_data_util.save_prediction_to_db(pred_data_ext, ds=ds, model_name=model_name, stage="prediction",
                                              pred_name=label_name, pred_val_col_name=label_name_pred,
                                              label_col_name=label_name,
                                              run_id=run_id)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', type=str, default="", help='biz date')
    parser.add_argument('--model_name', type=str, default="lstm", help='model name, can be ae_lstm or lstm, default lstm')
    parser.add_argument('--epoch_num', type=int, default=5, help='epoch num')
    parser.add_argument('--batch_size', type=int, default=1001, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=128, help='lstm hidden variable dimension')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--num_layers', type=int, default=2, help='lstm number of layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='lstm dropout probability')

    parser.add_argument('--use_roc_label', type=int, default=1, help='use_roc_label 0 no, 1 yes, default 1')
    parser.add_argument('--use_log_close', type=int, default=0, help='use log for close price, 0 no, 1 yes, default 0')
    parser.add_argument('--use_sqrt_roc', type=int, default=0, help='use sqrt for roc, 0 no, 1 yes, default 0')
    parser.add_argument('--use_categorical_features', type=int, default=1,
                        help='use_categorical_features, 0 no, 1 yes, default 1')
    parser.add_argument('--label_name', type=str, default="label_roi_3d",
                        help='if used, use_roc_label,use_sqrt_roc,use_log_close will be disabled.')

    parser.add_argument('--save_prediction_result', type=int, default=1,
                        help='save prediction result to db, 0 no, 1 yes, default 1')

    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = get_args()
    args.data_file_name = quant_data_util.generate_data_file_name(args.ds)

    logger.info("execute task on ds {}".format(args.ds))
    logger.info("arguments {}".format(args))

    if not is_trade_date(args.ds):
        logger.info(f"{args.ds} is not trade date. task exits.")
        exit(os.EX_OK)

    train(args)


