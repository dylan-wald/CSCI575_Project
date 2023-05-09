import numpy as np
import matplotlib.pyplot as plt

class plot_all():

    def __init__(self, X_train, y_train, Loss_all, 
                Y_pred_all, Y_true, Y_pred_test_all, 
                Y_true_test, cases, A_all=None, B_all=None, E_all=None):

        self.X_train = X_train
        self.y_train = y_train

        self.Loss_all = Loss_all

        self.Y_true = Y_true
        self.Y_pred_all = Y_pred_all

        self.Y_pred_test_all = Y_pred_test_all
        self.Y_true_test = Y_true_test

        self.A_all = A_all
        self.B_all = B_all
        self.E_all = E_all

        self.cases = cases

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

        self.msdot_vec = msdot_train_plot[16:]
        self.Tsa_vec = Tsa_train_plot[16:]
        self.Toa_vec = Toa_train_plot[16:]
        self.Qsolar_vec = Qsolar_train_plot[16:]
        self.Qint_vec = Qint_train_plot[16:]

        self.time_vec_train = np.linspace(0,10,y_train_plot.shape[0])

        fig, ax = plt.subplots(3,1)

        ax1 = ax[0]
        ax1.plot(self.time_vec_train, self.Tsa_vec, label="T_sa")
        ax1.plot(self.time_vec_train, self.msdot_vec, label="ms_dot")
        ax1.set_ylabel("C, kg/s")
        ax1.set_title("Inputs")
        ax1.legend()

        ax2 = ax[1]
        ax2.plot(self.time_vec_train, self.Toa_vec, label="T_oa")
        ax2.plot(self.time_vec_train, self.Qsolar_vec, label="Q_solar")
        ax2.plot(self.time_vec_train, self.Qint_vec, label="Q_int")
        ax2.set_ylabel("C, W, W")
        ax2.set_title("Disturbances")
        ax2.legend()

        ax3 = ax[2]
        ax3.plot(self.time_vec_train, np.array(y_train_plot).flatten(), label="T_z")
        ax3.set_title("Outputs")
        ax3.set_ylabel("C")
        ax3.set_xlabel("Days")
        ax3.legend()

        plt.tight_layout()

        if show:
            plt.show()

    def plot_loss(self, show=False):

        fig, ax = plt.subplots(1,1)

        for i, loss in enumerate(self.Loss_all):
            ax.plot(loss, label=self.cases[i])
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend()

        if show:
            plt.show()

    def plot_train_result(self, show=False):
        
        fig, ax = plt.subplots(3,1)

        # True y data
        y_true_plot = np.zeros(np.shape(self.Y_true[:, 0, :].detach().numpy().flatten()))
        for j in np.arange(10):
            for i in range(18):            
                y_true_plot = np.concatenate((y_true_plot, np.array(self.Y_true[:, j+(i*10), :].detach().numpy().flatten())))
        y_true_plot = y_true_plot[16:]

        time_vec_train = np.linspace(0,10,y_true_plot.shape[0])

        # Predicted y data for each model
        for k, Y_pred in enumerate(self.Y_pred_all):
            y_pred_plot = np.zeros(np.shape(Y_pred[:, 0, :].detach().numpy().flatten()))
            for j in np.arange(10):
                for i in range(18):            
                    y_pred_plot = np.concatenate((y_pred_plot, np.array(Y_pred[:, j+(i*10), :].detach().numpy().flatten())))
            y_pred_plot = y_pred_plot[17:]

            time_vec = np.linspace(0,10,y_pred_plot.shape[0])

            MSE = np.sum((y_pred_plot - y_true_plot[:-1])**2)/len(y_pred_plot)
            print("MSE for {}: {}".format(self.cases[k], MSE))

            ax[k].plot(time_vec_train, y_true_plot, "r--", label="True T_z")
            ax[k].plot(time_vec, y_pred_plot, "b-", label="Pred T_z")
            ax[k].set_ylabel("Zone Temp. [C]")
            ax[k].set_title(self.cases[k])
            if k == 2:
                ax[k].set_xlabel("Days")
            ax[k].legend()
        plt.tight_layout()

        plt.show()


        fig, ax = plt.subplots(2,1)

        k = 0
        for A, B, E in zip(self.A_all, self.B_all, self.E_all):
            if A is not None:
                y_pred_SSM_plot = []
                x = y_true_plot[0].reshape((1,1))
                for i in range(len(y_true_plot)):
                    u = np.array([[self.Tsa_vec[i]], [self.msdot_vec[i]]])
                    d = np.array([[self.Toa_vec[i]],[self.Qsolar_vec[i]],[self.Qint_vec[i]]])
                    x = np.dot(A, x) + np.dot(B, u) + np.dot(E, d)
                    y_pred_SSM_plot.append(x[0][0])

                ax[k].plot(time_vec_train, y_true_plot, "r--", label="True T_z")
                ax[k].plot(time_vec_train, y_pred_SSM_plot, "b-", label="Pred T_z")
                ax[k].set_ylabel("Zone Temp. [C]")
                ax[k].set_title(self.cases[k]+" SSM")
                if k == 1:
                    ax[k].set_xlabel("Days")
                ax[k].legend()

                MSE = np.sum((y_true_plot - y_pred_SSM_plot)**2)/len(y_true_plot)
                print("MSE for {} (SSM): {}".format(self.cases[k], MSE))

                print("A matrix for {}: {}".format(self.cases[k], A))
                print("B matrix for {}: {}".format(self.cases[k], B))
                print("E matrix for {}: {}".format(self.cases[k], E))

                k=k+1
        plt.tight_layout()

        # if show:
        plt.show()


    def plot_test_result(self, show=False):
        
        fig, ax = plt.subplots(3,1)

        # True y data
        Y_true_test_plot = np.zeros(np.shape(self.Y_true_test[:, 0, :].detach().numpy().flatten()))
        for j in np.arange(4):
            for i in range(18):            
                Y_true_test_plot = np.concatenate((Y_true_test_plot, np.array(self.Y_true_test[:, j+(i*4), :].detach().numpy().flatten())))
        Y_true_test_plot = Y_true_test_plot[16:]

        time_vec_test = np.linspace(0,4,Y_true_test_plot.shape[0])

        # Predicted y data for each model
        for k, Y_pred_test in enumerate(self.Y_pred_test_all):
            y_pred_test_plot = np.zeros(np.shape(Y_pred_test[:, 0, :].detach().numpy().flatten()))
            for j in np.arange(4):
                for i in range(18):            
                    y_pred_test_plot = np.concatenate((y_pred_test_plot, np.array(Y_pred_test[:, j+(i*4), :].detach().numpy().flatten())))
            y_pred_test_plot = y_pred_test_plot[17:]

            time_vec = np.linspace(0,4,y_pred_test_plot.shape[0])

            MSE = np.sum((y_pred_test_plot - Y_true_test_plot[:-1])**2)/len(y_pred_test_plot)
            print("MSE for {}: {}".format(self.cases[k], MSE))

            ax[k].plot(time_vec_test, Y_true_test_plot, "r--", label="True T_z")
            ax[k].plot(time_vec, y_pred_test_plot, "b-", label="Pred T_z")
            ax[k].set_ylabel("Zone Temp. [C]")
            ax[k].set_title(self.cases[k])
            if k == 2:
                ax[k].set_xlabel("Days")
            ax[k].legend()
        plt.tight_layout()

        plt.show()


        fig, ax = plt.subplots(2,1)

        k = 0
        for A, B, E in zip(self.A_all, self.B_all, self.E_all):
            if A is not None:
                y_pred_test_SSM_plot = []
                x = Y_true_test_plot[0].reshape((1,1))
                for i in range(len(Y_true_test_plot)):
                    u = np.array([[self.Tsa_vec[i]], [self.msdot_vec[i]]])
                    d = np.array([[self.Toa_vec[i]],[self.Qsolar_vec[i]],[self.Qint_vec[i]]])
                    x = np.dot(A, x) + np.dot(B, u) + np.dot(E, d)
                    y_pred_test_SSM_plot.append(x[0][0])

                ax[k].plot(time_vec_test, Y_true_test_plot, "r--", label="True T_z")
                ax[k].plot(time_vec_test, y_pred_test_SSM_plot, "b-", label="Pred T_z")
                ax[k].set_ylabel("Zone Temp. [C]")
                ax[k].set_title(self.cases[k]+" SSM")
                if k == 1:
                    ax[k].set_xlabel("Days")
                ax[k].legend()

                MSE = np.sum((Y_true_test_plot - y_pred_test_SSM_plot)**2)/len(Y_true_test_plot)
                print("MSE for {} (SSM): {}".format(self.cases[k], MSE))

                k=k+1
        plt.tight_layout()

        # if show:
        plt.show()