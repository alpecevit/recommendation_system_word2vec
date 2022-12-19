from gensim.models.callbacks import CallbackAny2Vec


class loss_calculator(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1
        self.losses = []
        self.total_losses = []

    def on_epoch_end(self, model):
        cumulative_loss = model.get_latest_training_loss()
        loss = cumulative_loss if self.epoch <= 1 else cumulative_loss - self.total_losses[-1]
        print(f"Loss after epoch {self.epoch}: {loss} / Cumulative loss: {cumulative_loss}")
        self.epoch += 1
        self.losses.append(loss)
        self.total_losses.append(cumulative_loss)