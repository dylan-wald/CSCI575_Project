import numpy as np
import matplotlib.pyplot as plt

class plotting():

    def __init__(self, X_train, y_train, Loss, Y_pred, Y_true, Y_pred_test, Y_true_test, A=None, B=None, E=None):

        self.X_train = X_train
        self.y_train = y_train

        self.Loss = Loss

        self.Y_true = Y_true
        self.Y_pred = Y_pred

        self.Y_pred_test = Y_pred_test
        self.Y_true_test = Y_true_test

        self.A = A
        self.B = B
        self.E = E

        self.plot_train(show=True)
        self.plot_loss(show=True)
        self.plot_train_result(show=True)
        self.plot_test_result(show=True)
    
    def plot_train(self, show=False):

        y_train_plot = np.zeros(np.shape(self.y_train[:, 0, :].detach().numpy().flatten()))

        msdot_train_plot = np.zeros(np.shape(self.X_train[:, 0, 1].detach().numpy().flatten()))
        Tsa_train_plot = np.zeros(np.shape(self.X_train[:, 0, 1].detach().numpy().flatten()))
        Toa_train_plot = np.zeros(np.shape(self.X_train[:, 0, 1].detach().numpy().flatten()))
        Qsolar_train_plot = np.zeros(np.shape(self.X_train[:, 0, 1].detach().numpy().flatten()))
        Qint_train_plot = np.zeros(np.shape(self.X_train[:, 0, 1].detach().numpy().flatten()))
        for j in np.arange(10):
            for i in range(18):
                x_train_plot = np.array(self.X_train[:, j+(i*10), :].detach().numpy())
                msdot_train_plot = np.concatenate((msdot_train_plot, x_train_plot[:,1]))
                Tsa_train_plot = np.concatenate((Tsa_train_plot, x_train_plot[:,0]))
                Toa_train_plot = np.concatenate((Toa_train_plot, x_train_plot[:,2]))
                Qsolar_train_plot = np.concatenate((Qsolar_train_plot, x_train_plot[:,3]))
                Qint_train_plot = np.concatenate((Qint_train_plot, x_train_plot[:,4]))
                y_train_plot = np.concatenate((y_train_plot, np.array(self.y_train[:, j+(i*10), :].detach().numpy().flatten())))
        y_train_plot = np.array(y_train_plot).reshape((2896,1))
        y_train_plot = y_train_plot[16:]

        self.msdot_vec = msdot_train_plot
        self.Tsa_vec = Tsa_train_plot
        self.Toa_vec = Toa_train_plot
        self.Qsolar_vec = Qsolar_train_plot
        self.Qint_vec = Qint_train_plot

        fig, ax = plt.subplots(3,1)

        ax1 = ax[0]
        ax1.plot(Tsa_train_plot, label="T_sa")
        ax1.plot(msdot_train_plot, label="ms_dot")
        ax1.set_title("Inputs")
        ax1.legend()

        ax2 = ax[1]
        ax2.plot(Toa_train_plot, label="T_oa")
        ax2.plot(Qsolar_train_plot, label="Q_solar")
        ax2.plot(Qint_train_plot, label="Q_int")
        ax2.set_title("Disturbances")
        ax2.legend()

        ax3 = ax[2]
        ax3.plot(np.array(y_train_plot).flatten(), label="T_z")
        ax3.set_title("Outputs")
        ax3.legend()

        if show:
            plt.show()

    def plot_loss(self, show=False):

        fig, ax = plt.subplots(1,1)

        ax.plot(self.Loss)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")

        if show:
            plt.show()

    def plot_train_result(self, show=False):

        y_true_plot = np.zeros(np.shape(self.Y_true[:, 0, :].detach().numpy().flatten()))
        y_pred_plot = np.zeros(np.shape(self.Y_pred[:, 0, :].detach().numpy().flatten()))
        for j in np.arange(10):
            for i in range(18):            
                y_true_plot = np.concatenate((y_true_plot, np.array(self.Y_true[:, j+(i*10), :].detach().numpy().flatten())))
                y_pred_plot = np.concatenate((y_pred_plot, np.array(self.Y_pred[:, j+(i*10), :].detach().numpy().flatten())))

        y_true_plot = y_true_plot[16:]
        y_pred_plot = y_pred_plot[16:]

        if self.A is not None:
            y_pred_SSM_plot = []
            x = y_true_plot[0].reshape((1,1))
            for i in range(len(y_true_plot)):
                u = np.array([[self.Tsa_vec[i+15]], [self.msdot_vec[i+15]]])
                d = np.array([[self.Toa_vec[i+15]],[self.Qsolar_vec[i+15]],[self.Qint_vec[i+15]]])
                x = np.dot(self.A, x) + np.dot(self.B, u) + np.dot(self.E, d)
                y_pred_SSM_plot.append(x[0][0])

        fig, ax = plt.subplots(1,1)

        ax.plot(y_true_plot, label="True T_z")
        ax.plot(y_pred_plot, label="Pred T_z")
        if self.A is not None:
            ax.plot(y_pred_SSM_plot, label="Pred T_z (SSM)")
        ax.set_title("Train Data")
        ax.legend()

        if show:
            plt.show()

    def plot_test_result(self, show=False):

        Y_pred_test_plot = np.zeros(np.shape(self.Y_pred_test[:, 0, :].detach().numpy().flatten()))
        Y_true_test_plot = np.zeros(np.shape(self.Y_true_test[:, 0, :].detach().numpy().flatten()))
        for j in np.arange(4):
            for i in range(18):            
                Y_true_test_plot = np.concatenate((Y_true_test_plot, np.array(self.Y_true_test[:, j+(i*4), :].detach().numpy().flatten())))
                Y_pred_test_plot = np.concatenate((Y_pred_test_plot, np.array(self.Y_pred_test[:, j+(i*4), :].detach().numpy().flatten())))

        Y_true_test_plot = Y_true_test_plot[16:]
        Y_pred_test_plot = Y_pred_test_plot[16:]

        if self.A is not None:
            y_pred_test_SSM_plot = []
            x = Y_true_test_plot[0].reshape((1,1))
            for i in range(len(Y_true_test_plot)):
                u = np.array([self.Tsa_vec[i+15], self.msdot_vec[i+15]])
                d = np.array([self.Toa_vec[i+15],self.Qsolar_vec[i+15],self.Qint_vec[i+15]])
                x = np.dot(self.A, x) + np.dot(self.B, u) + np.dot(self.E, d)
                y_pred_test_SSM_plot.append(x[0][0])

        fig, ax = plt.subplots(1,1)

        ax.plot(Y_true_test_plot, label="True T_z")
        ax.plot(Y_pred_test_plot, label="Pred T_z")
        if self.A is not None:
            ax.plot(y_pred_test_SSM_plot, label="Pred T_z (SSM)")
        ax.set_title("Test Data")
        ax.legend()

        if show:
            plt.show()