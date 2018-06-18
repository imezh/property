from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt
import dbconfig as cfg

conn = 'mysql+mysqlconnector://' + cfg.mysql['user'] + ':' + cfg.mysql['password'] + '@' + cfg.mysql['host'] + '/' + \
       cfg.mysql['db'] + ''

engine = create_engine(conn)

Session = sessionmaker(bind=engine)

session = Session()

Base = declarative_base()


class Point(Base):
    __tablename__ = 'Points_samp1000'
    ID = Column(Integer, primary_key=True)
    Latitude = Column(Float)
    Longitude = Column(Float)



points = session.query(Point).all()

x = []
y = []

for Point in points:
    print(Point.Latitude, Point.Longitude)
    x.append(Point.Longitude)
    y.append(Point.Latitude)


coords = np.column_stack([x, y])

y_pred = KMeans(n_clusters=4, random_state=0).fit_predict(coords)
plt.subplot(1, 1, 1)
plt.scatter(x, y, c=y_pred)
plt.show()
