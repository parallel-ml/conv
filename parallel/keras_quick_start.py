from keras.models import load_model
from keras.utils import plot_model

model = load_model('/home/jiashenc/0000_epoch-4.0079_loss-0.0253_acc-4.1435_val_loss-0.0266_val_acc.hdf5')

print model.summary()
