from keras import backend as K

def get_weakly_supervised_sparse_categorical_crossentropy(num_classes, alpha=0.1):
    def weakly_supervised_sparse_categorical_crossentropy(y_true, y_pred):
        y_true = y_true[:,:,:,0] # squeeze
        mask = K.cast(y_true < num_classes, K.floatx()) # annotated pixels
        inv_mask = K.cast(y_true >= num_classes, K.floatx()) # non-annotated pixels
        y_true = y_true - inv_mask * num_classes # offset non-annotated pixels
        s = mask * K.sparse_categorical_crossentropy(y_true, y_pred)
        n = inv_mask * K.sparse_categorical_crossentropy(y_true, y_pred)
        return s - alpha * n
    return weakly_supervised_sparse_categorical_crossentropy

def get_semisupervised_sparse_categorical_crossentropy(num_classes, alpha=0.1):
    def semisupervised_sparse_categorical_crossentropy(y_true, y_pred):
        y_true = y_true[:,:,:,0] # squeeze
        mask = K.cast(y_true < num_classes, K.floatx()) # annotated pixels
        inv_mask = K.cast(y_true >= num_classes, K.floatx()) # non-annotated pixels
        y_true = y_true - inv_mask * num_classes # offset non-annotated pixels
        s = mask * K.sparse_categorical_crossentropy(y_true, y_pred)
        n = inv_mask * K.categorical_crossentropy(y_pred, y_pred)
        return s + alpha * n
    return semisupervised_sparse_categorical_crossentropy
