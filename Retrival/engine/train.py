from ignite.engine import Events,create_supervised_trainer,create_supervised_evaluator
from ignite.handlers import Timer,TerminateOnNan,ModelCheckpoint
from ignite.metrics import Loss,RunningAverage,Accuracy
def train(train_loader, val_loader,model, criterion, optimizer, epoch,writer,log_interval):
    trainer=create_supervised_trainer(model,optimizer,criterion)
    valMetrics={
        'accuracy':Accuracy(),
        'loss':Loss(criterion)
    }
    evaluator=create_supervised_evaluator(model,metrics=valMetrics)

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(trainer):
        print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics["accuracy"], metrics["nll"]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics["accuracy"], metrics["nll"]))

    trainer.run(train_loader, max_epochs=epoch)