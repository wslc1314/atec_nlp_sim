# from models.SenMatchSen import Config,Model
# from models.SeqMatchSeq import Config,Model
from models.SeqMatchSeq_BiMPM import Config,Model
import tensorflow as tf
import glob

flags = tf.app.flags
flags.DEFINE_bool('focal',False,'Whether to use focal loss.')
flags.DEFINE_float('alpha', 0.75 ,'Alpha value for focal loss.')
flags.DEFINE_float('dropout', 0.1 ,'Value for dropout.')
flags.DEFINE_bool('fine_tune',False,'Choose to fine-tune or train.')
flags.DEFINE_float('init_learning_rate', 0.001,'Value for initial learning rate.')
flags.DEFINE_bool('with_validation',True,'Whether to go training with validation.')
flags.DEFINE_integer('max_to_save',3,'Maximum to save.')
flags.DEFINE_integer('num_epochs',500,'Total epochs.')
flags.DEFINE_integer('steps_every_epoch',100,'Steps in an epoch.')
flags.DEFINE_integer('batch_size',128,"Size of a batch.")
flags.DEFINE_integer('save_epochs',10,'Save epochs.')
flags.DEFINE_integer('early_stopping',10,"Metric for early stopping.")
flags.DEFINE_integer('epoch_adam_to_sgd',501,"Start epoch for changing adam to sgd.")
FLAGS = flags.FLAGS


def main(_):

    Config.focal=FLAGS.focal
    Config.alpha=FLAGS.alpha
    Config.dropout=FLAGS.dropout
    Config.fine_tune=FLAGS.fine_tune
    Config.init_learning_rate=FLAGS.init_learning_rate
    model=Model(Config)

    log_dir="logs/SeqMatchSeq_BiMPM/1st"
    save_dir=log_dir.replace("logs","checkpoints")
    load_dir=None
    model.fit(with_validation=FLAGS.with_validation,
              load_path=load_dir,log_dir=log_dir,save_dir=save_dir,max_to_keep=FLAGS.max_to_save,
              num_epochs=FLAGS.num_epochs,steps_every_epoch=FLAGS.steps_every_epoch,
              batch_size=FLAGS.batch_size,save_epochs=FLAGS.save_epochs,
              early_stopping=FLAGS.early_stopping,epoch_adam_to_sgd=FLAGS.epoch_adam_to_sgd)
    load_dir=save_dir+"/trainval"
    for load_path in glob.glob(load_dir+"/*.meta"):
        load_path=load_path.replace(".meta","")
        model.evaluate(validFile="data/atec/10/train0.csv",dictPath="data/atec/10/train0_dict.json",
                       load_path=load_path)
        model.evaluate(validFile="data/atec/10/valid0.csv",dictPath="data/atec/10/train0_dict.json",
                       load_path=load_path)


if __name__=="__main__":
    tf.app.run()
