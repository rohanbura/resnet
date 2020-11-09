import pandas as pd

model = load_model('inception.h5')

test_ds.reset()

pred = model.predict_generator(test_ds,verbose=1)

predicted_class_indices = np.argmax(pred,axis=1)

labels = (train_set.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]
filenames = test_ds.filenames
filen = [name[5:] for name in filenames]
results = pd.DataFrame({"filename":filen,
                      "class":predictions})

results.to_csv("output1.csv",index=False)