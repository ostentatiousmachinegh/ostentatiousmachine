
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score


class LinearRegression(torch.nn.Module):
    def __init__(self,learning_rate=0.01,epochs=1000):
        super().__init__()
        self.learning_rate=learning_rate
        self.epochs=epochs

        self.w0=torch.nn.Parameter(torch.tensor(0.0))
        self.w1=torch.nn.Parameter(torch.tensor(0.0))

        self.optimizer=torch.optim.SGD(self.parameters(),lr=self.learning_rate        )
        self.loss_function=torch.nn.MSELoss()

        self.w0_history=[]
        self.w1_history=[]
        self.loss_history=[]

    @staticmethod
    def _tensor(values):
        if isinstance(values,torch.Tensor):
            return values.detach().to(dtype=torch.float32).reshape(-1)
        return torch.as_tensor(np.asarray(values,dtype=np.float32).reshape(-1)        )

    def forward(self,x):
        return self.w0+self.w1*x

    def fit(self,X_train,y_train,X_test,y_test):
        x_train=self._tensor(X_train)
        target_train=self._tensor(y_train)
        x_test=self._tensor(X_test)
        target_test=self._tensor(y_test)

        if len(x_train)!=len(target_train):
            raise ValueError("training x and y lengths differ.")
        if len(x_test)!=len(target_test):
            raise ValueError("testing x and y lengths differ.")

        self.w0_history.clear()
        self.w1_history.clear()
        self.loss_history.clear()
        self.train()

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()

            predictions=self.forward(x_train)
            loss=self.loss_function(predictions,target_train)
            loss.backward()
            self.optimizer.step()

            self.w0_history.append(self.w0.item())
            self.w1_history.append(self.w1.item())
            self.loss_history.append(loss.item())

        self.eval()

        with torch.no_grad():
            test_predictions=self.forward(x_test)

        self.test_r2=r2_score(target_test.numpy(),test_predictions.numpy())

        print("training completed")
        print(f"w0 (intercept): {self.w0.item():.6f}")
        print(f"w1 (slope): {self.w1.item():.6f}")
        print(f"Test R-squared: {self.test_r2:.6f}")

        return self

    def predict(self,BCR):
        self.eval()

        x=self._tensor(BCR)

        with torch.no_grad():
            predictions=self.forward(x)

        return predictions.numpy()

    def analysis_plot(self,X,y):
        x=self._tensor(X).numpy()
        target=self._tensor(y).numpy()

        x_line=np.linspace(x.min(),x.max(),200)
        y_line=self.predict(x_line)

        fig,axes=plt.subplots(2,2,figsize=(13,9))

        axes[0,0].scatter(x,target,alpha=0.6,label="original data")
        axes[0,0].plot(x_line,y_line,color="red",label="regression line")
        axes[0,0].set_title("BCR vs annual production")
        axes[0, 0].set_xlabel("BCR")
        axes[0, 0].set_ylabel("annual production")
        axes[0, 0].legend()

        axes[0,1].plot(self.w0_history)
        axes[0,1].set_title("w0 during training")
        axes[0,1].set_xlabel("epoch")

        axes[1,0].plot(self.w1_history)
        axes[1,0].set_title("w1 during training")
        axes[1,0].set_xlabel("epoch")

        axes[1,1].plot(self.loss_history)
        axes[1,1].set_title("training loss")
        axes[1,1].set_xlabel("epoch")

        for ax in axes.flat:
            ax.grid(True)

        plt.tight_layout()
        plt.show()