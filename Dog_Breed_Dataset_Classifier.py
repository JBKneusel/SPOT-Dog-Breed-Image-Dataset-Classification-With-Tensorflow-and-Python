import os
import sys
from sklearn.metrics import matthews_corrcoef
import tensorflow.keras as keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import matplotlib.pylab as plt

##### ESTABLISHING LAYERS AND INPUT:

def Confusion_Matrix(pred, y, num_classes=10):
    ### This returns a basic confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype = "uint16")
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1
    acc = np.diag(cm).sum() / cm.sum()
    return cm, acc
    ### 6x6 matrix for 6 classes, tallying predicted label and all possible combinations, pred and y are vectors.

def Conv_Block(_, filters):
    ### This is a VGG Convolutional-relu-maxpooling block with dropout
    _ = Conv2D(filters, (3,3), padding='same')(_)
    _ = BatchNormalization()(_)
    _ = ReLU()(_)
    _ = Conv2D(filters, (3,3), padding='same')(_)
    _ = BatchNormalization()(_)
    _ = ReLU()(_)
    return MaxPooling2D((2,2))(_)
    ### 2D convulutional layer for dealing with images, ReLU non-linearity, makes negative values 0. 2D pooling, takes the output tensor and reduces width and height by a factor of two 
    ### Keeping max values in each region 

def Dense_Block(_, nodes):
    ### This is a fully connected layer block with a dense layer, batch normalization, ReLU activation, and dropout. 
    _ = Dense(nodes)(_) ### Standard feed forward network layer. Old school multi-layer perceptron (traditional neural net)
    _ = BatchNormalization()(_)
    _ = ReLU()(_)
    return Dropout(0.5)(_)

def VGG(input_shape, num_classes=10):
    ###This returns an instance of the VGG8 Model 
    inp = Input(input_shape)
    _ = Conv_Block(inp,64) ### Learn three convolutional blocks (6 layers) increasing filters. 
    _ = Conv_Block(_, 128)
    _ = Conv_Block(_, 256)
    _ = Flatten()(_)
    _ = Dense_Block(_, 2048) 
    _ = Dense_Block(_, 2048)
    _ = Dense(num_classes)(_)
    output = Softmax()(_)
    return Model(inputs=inp, outputs=output)

    ### Con blocks learn a new representation of the input image, dense blocks are traditional neural nets. 
    ### Softmax transforms vector into a sudo probability distribution. Every value in the ouput vector is a 
    ### value between 0-1 that represents how strongly the model believes the input belongs to the respective class
    ### label prediction is the index of the largest softmax value. the largest softmax value is used as the label
    ### since we have known labels we can construct a confusion matrix to know how well we are doing.

    ### Classes:


    #  Command line
if (len(sys.argv) == 1):
    print()
    print("Classifiers <minibatch> <epochs> <outdir> [<nsamp>]")
    print()
    print("  <minibatch> - minibatch size (e.g. 128)")
    print("  <epochs>    - number of training epochs (e.g. 16)")
    print("  <outdir>    - output file directory (overwritten)")
    print("  <nsamp>     - number of training samples (optional, all if not given)")
    print()
    exit(0)

#### MODEL IMPLEMENTATION

### Reading command line
batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
outdir = sys.argv[3]
nsamp = 50000 if len(sys.argv) < 5 else int(sys.argv[4]) ### Optional samples to use

### additional params
num_classes = 6
img_rows, img_cols = 64, 64
input_shape = (img_rows, img_cols, 3)

### Load bird image datasets:
x_train = np.load("bird6_64_xtrain.npy")[:nsamp]
y_train = np.load("bird6_ytrain.npy")[:nsamp]
x_test = np.load("bird6_64_xtest.npy")
y_test = np.load("bird6_ytest.npy")

### Scale [0,1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

### Converting labels into vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
ytest = y_test.copy()
y_test = keras.utils.to_categorical(y_test, num_classes)

### Keep training samples
N = int(0.1*len(x_train))
x_val, x_train = x_train[:N], x_train[N:]
y_val, y_train = y_train[:N], y_train[N:]

### building the model
model = VGG(input_shape, num_classes=num_classes)
model.summary()

###Compilation and training
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

##Testing 
pred = model.predict(x_test, verbose=0)
plabel = np.argmax(pred, axis=1)
cm, acc = Confusion_Matrix(plabel, ytest, num_classes=num_classes)
mcc = matthews_corrcoef(ytest, plabel)
txt = "Test set accuracy: %0.4f, MCC: %0.4f" % (acc,mcc)
print(cm)
print(txt)
print()

os.system("mkdir %s 2>/dev/null" % outdir) 
np.save(outdir+"/confusion_matrix.npy", cm)
model.save(outdir+"/model.keras")

### Plot Training vs Validation Error
train_error = 1.0 - np.array(history.history["accuracy"])
val_error = 1.0 - np.array(history.history["val_accuracy"])
epochs_range = list(range(epochs))

with open(outdir+"/accuracy_mcc.txt","w") as f:
    f.write(txt+"\n")
np.save(outdir+"/confusion_matrix.npy", cm)
model.save(outdir+"/model.keras")
terr = 1.0 - np.array(history.history['accuracy'])
verr = 1.0 - np.array(history.history['val_accuracy'])
x = list(range(epochs))
plt.plot(x, terr, linestyle='solid', linewidth=0.5, color='k', label='train')
plt.plot(x, verr, linestyle='solid', color='k', label='validation')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig(outdir+"/error_plot.png", dpi=300)
