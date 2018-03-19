import os
import numpy as np
from pickle import load
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.models import Model, Input
from keras.layers import Dense, GlobalMaxPooling2D, RepeatVector, Embedding, LSTM, TimeDistributed, concatenate
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def prepare_data():
    # get image ids
    filenames = {'train': 'Flickr8k_text/Flickr_8k.trainImages.txt',
                 'dev': 'Flickr8k_text/Flickr_8k.devImages.txt',
                 'test': 'Flickr8k_text/Flickr_8k.testImages.txt'}
    image_ids = {}
    for key, filename in filenames.items():
        file = open(filename, 'r')
        ids = file.read().split()
        image_ids[key] = ids
        file.close()

    # load image description
    image_descriptions = {}
    filename = 'Flickr8k_text/Flickr8k.token.txt'
    file = open(filename, 'r')
    lines = file.read().split('\n')
    train_descriptions = {}
    dev_descriptions = {}
    test_descriptions = {}
    for line in lines:
        split_list = line.replace('#', '\t').split('\t')
        if len(split_list)>=3 and split_list[1] == '0':
            id = split_list[0]
            description = 'startseq ' + ''.join(split_list[2:]) + ' endseq'
            if id in image_ids['train']:
                train_descriptions[id] = description
            elif id in image_ids['dev']:
                dev_descriptions[id] = description
            elif id in image_ids['test']:
                test_descriptions[id] = description
    file.close()
    image_descriptions['train'] = train_descriptions
    image_descriptions['dev'] = dev_descriptions
    image_descriptions['test'] = test_descriptions
    print(image_descriptions)

    # load image features
    image_features = {}
    filename = 'features.pkl'
    features = load(open(filename, 'rb'))
    train_features = {}
    dev_features = {}
    test_features = {}
    for id, feature in features.items():
        if id in image_ids['train']:
            train_features[id] = feature
        elif id in image_ids['dev']:
            dev_features[id] = feature
        elif id in image_ids['test']:
            test_features[id] = feature
    file.close()
    image_features['train'] = train_features
    image_features['dev'] = dev_features
    image_features['test'] = test_features
    #print(image_features)
    return image_features, image_descriptions


def descriptions_summary(image_descriptions):
    description_lengths = [len(description) for description in image_descriptions]
    max_length = max(description_lengths)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([description for description in image_descriptions.values()])
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, max_length, vocab_size


def data_generator(image_features, image_descriptions, tokenizer, max_length, vocab_size, images_per_epoch, n_images):
    x_images, x_text, y = list(), list(), list()
    # integer encode the description
    while True:
        random_ids = np.random.choice(list(image_descriptions.keys()), n_images, replace=False)
        for index, id in enumerate(random_ids):
            if index%images_per_epoch == 0:
                x_images, x_text, y = list(), list(), list()
            description = image_descriptions[id]
            feature = image_features[id][0]
            #print(description)
            seq = tokenizer.texts_to_sequences([description])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                x_images.append(feature)
                x_text.append(in_seq)
                y.append(out_seq)
            if index%images_per_epoch == 0:
                yield [[np.asarray(x_images), np.asarray(x_text)], np.asarray(y)]


def build_model(vocab_size, max_length):
    # feature extractor (encoder)
    inputs1 = Input(shape=(7, 7, 512))
    fe1 = GlobalMaxPooling2D()(inputs1)
    fe2 = Dense(128, activation='relu')(fe1)
    fe3 = RepeatVector(max_length)(fe2)
    # embedding
    inputs2 = Input(shape=(max_length,))
    emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
    emb3 = LSTM(256, return_sequences=True)(emb2)
    emb4 = TimeDistributed(Dense(128, activation='relu'))(emb3)
    # merge inputs
    merged = concatenate([fe3, emb4])
    # language model (decoder)
    lm2 = LSTM(500)(merged)
    lm3 = Dense(500, activation='relu')(lm2)
    outputs = Dense(vocab_size, activation='softmax')(lm3)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    #plot_model(model, show_shapes=True, to_file='plot.png')
    return model


def generate_description(model, tokenizer, feature, max_length):
    description = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([description])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        y = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(y)
        id_to_word = {index: word for word, index in tokenizer.word_index.items()}
        word = id_to_word[yhat]
        if word is None:
            break
        description += ' ' + word
        if word == 'endseq':
            break
    return description


def evaluate_model(image_features, image_descriptions):
    predicted = []
    actural = []
    for id, feature in image_features.items():
        predict_description = generate_description(model, tokenizer, feature, max_length)
        actural_description = image_descriptions[id]
        predicted.append(predict_description)
        actural.append(actural_description)
    chencherry = SmoothingFunction()
    bleu = corpus_bleu(actural, predicted, smoothing_function=chencherry.method4)
    return bleu


if __name__ == '__main__':
    # prepare data
    image_features, image_descriptions = prepare_data()
    tokenizer, max_length, vocab_size = descriptions_summary(image_descriptions['train'])

    # train model
    n_epochs = 20
    n_images = 800
    images_per_epoch = 2
    #train_batches_per_epoch = int(len(image_descriptions['train']) / images_per_epoch)
    #dev_batches_per_epoch = int(len(image_descriptions['dev']) / images_per_epoch)
    train_batches_per_epoch = int(n_images / images_per_epoch)
    dev_batches_per_epoch = int(n_images / images_per_epoch)

    if os.path.exists('model.h5'):
        model = load_model('model.h5')
        print('Continue training ...')
    else:
        model = build_model(vocab_size, max_length)
    for j in range(500):
        train_generator = data_generator(image_features['train'], image_descriptions['train'],
                                         tokenizer, max_length, vocab_size, images_per_epoch, n_images)
        dev_generator = data_generator(image_features['dev'], image_descriptions['dev'],
                                       tokenizer, max_length, vocab_size, images_per_epoch, n_images)
        model.fit_generator(train_generator, steps_per_epoch=train_batches_per_epoch, epochs=n_epochs, verbose=2,
                        validation_data=dev_generator, validation_steps=dev_batches_per_epoch)
        model.save('model.h5')

        # evaluate model
        #bleu = evaluate_model(image_features['test'], image_descriptions['test'])
        #print('BLEU score of the model: {}'.format(bleu))

        # plot
        for i in range(30):
            id = list(image_features['test'].keys())[i]
            feature = image_features['test'][id]
            predict_description = generate_description(model, tokenizer, feature, max_length)
            true_description = image_descriptions['test'][id]
            print('True: {}\nPredicted: {}'.format(true_description, predict_description))
            #path = 'Flickr8k_Dataset/' + id
            #img = mpimg.imread(path)
            #imgplot = plt.imshow(img)
            #plt.show()
