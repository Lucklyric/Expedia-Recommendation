# coding: utf-8

# In[2]:

import tensorflow as tf
import pandas as pd
import numpy as np

# Prepare for latent variables
des_latent = pd.read_csv("../../data/destinations.csv")
np.save("../../data/destinations", des_latent)

# Prepare for records
chunksize = 100000

# In[3]:

train = pd.read_csv("../data/train.csv", iterator=True, chunksize=chunksize, na_values={'null', 'NaN'},
                    keep_default_na=True)

# In[4]:

writer_13 = tf.python_io.TFRecordWriter("../data/train-13.tfrecords")
writer_14 = tf.python_io.TFRecordWriter("../data/train-14.tfrecords")

# In[5]:

test = None
counter = 0
for train_chunk in train:
    counter += 1
    print ("Process %d, chunk" % counter)
    # Convert date_time
    train_chunk["srch_ci"] = pd.to_datetime(train_chunk["srch_ci"], errors="coerce")
    train_chunk["srch_co"] = pd.to_datetime(train_chunk["srch_co"], errors="coerce")
    train_chunk["date_time"] = pd.to_datetime(train_chunk["date_time"], errors="coerce")

    train_chunk = train_chunk.dropna(subset=["srch_ci", "srch_co", "date_time"])

    train_chunk["srch_ci_month"] = train_chunk["srch_ci"].dt.month
    train_chunk["srch_ci_day"] = train_chunk["srch_ci"].dt.day

    train_chunk["srch_co_month"] = train_chunk["srch_co"].dt.month
    train_chunk["srch_co_day"] = train_chunk["srch_co"].dt.day

    train_chunk["month"] = train_chunk["date_time"].dt.month
    train_chunk["year"] = train_chunk["date_time"].dt.year

    # Drop date_time
    train_chunk = train_chunk.drop(['date_time', 'srch_co', 'srch_ci'], axis=1)

    # Fill NaN
    train_chunk = train_chunk.fillna(-1)

    # Seperate to two years
    t1 = train_chunk[((train_chunk.year == 2013) | ((train_chunk.year == 2014) & (train_chunk.month < 8)))]
    t2 = train_chunk[((train_chunk.year == 2014) & (train_chunk.month >= 8))]

    # Drop year
    t1 = t1.drop(['year'], axis=1)
    t2 = t2.drop(['year'], axis=1)

    # Re-order
    cols = t1.columns.tolist()
    cols = cols[-5:] + cols[:-5]
    t1 = t1[cols]
    t2 = t2[cols]

    # Write to TFRecords
    for row in t1.values:
        example = tf.train.Example(features=tf.train.Features(
            feature={
                "label": tf.train.Feature(float_list=tf.train.FloatList(value=[row[-1]])),
                "feature": tf.train.Feature(float_list=tf.train.FloatList(value=row[:-1]))
            }
        ))
        writer_13.write(example.SerializeToString())

    for row in t2.values:
        example = tf.train.Example(features=tf.train.Features(
            feature={
                "label": tf.train.Feature(float_list=tf.train.FloatList(value=[row[-1]])),
                "feature": tf.train.Feature(float_list=tf.train.FloatList(value=row[:-1]))
            }
        ))
        writer_14.write(example.SerializeToString())
    # test = t2
    break
writer_13.close()
writer_14.close()

# In[6]:

print ("Done!!")


# In[ ]:
