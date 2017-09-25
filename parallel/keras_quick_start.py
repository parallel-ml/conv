from keras.models import load_model
from keras.utils import plot_model

model = load_model('/home/jiashen/weights/batch_4_noaug/199_epoch-0.2510_loss-0.9403_acc-6.5269_val_loss-0.3061_val_acc.hdf5')

print model.summary()
