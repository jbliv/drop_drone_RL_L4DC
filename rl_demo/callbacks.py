import wandb
from stable_baselines3.common.callbacks import BaseCallback



class WandBVideoCallback(BaseCallback):
    def __init__(self, verbose=0) -> None:
        super(WandBVideoCallback, self).__init__(verbose)
        #self.counter = 0

    def _on_step(self) -> bool:
        if self.training_env.plot is not None and not self.training_env.plot_uploaded:
            wandb.log({"plot": wandb.Image(self.training_env.plot)})
            self.training_env.uploaded(True)
        #    self.counter = 0    
        #self.counter += 1
        # if self.training_env.plot is not None and self.counter > 100 :
        #     wandb.log({"plot": wandb.Image(self.training_env.plot)})
        #     print("plot callback")
        #     self.counter = 0    
        # self.counter += 1
        return True