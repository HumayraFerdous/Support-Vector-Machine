import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

X = np.array([[1,1],[2,2],[4,3],[3,4]])
y = np.array([1,1,-1,-1])

clf = svm.SVC(kernel ='linear',C = 1000)
clf.fit(X,y)

plt.scatter(X[:,0],X[:,1],c=y,cmap='bwr',s=100)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0],xlim[1],40)
yy = np.linspace(ylim[0],ylim[1],30)
YY, XX = np.meshgrid(yy,xx)
xy = np.vstack([XX.ravel(),YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1,0, 1],
           linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0],
           clf.support_vectors_[:, 1],
           s=200, linewidth=1, facecolors='none', edgecolors='k')

plt.xlabel('Width')
plt.ylabel('Height')
plt.title('Linear SVM: Squares vs. Circles')
plt.show()

print(clf.predict([[2,1]]))
