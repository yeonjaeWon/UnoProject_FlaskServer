from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey


Base = declarative_base()

class Unofarm(Base):
    __tablename__ = "FARM_CONTROL"

    DNO = Column(Integer, primary_key=True)
    D_LABEL = Column(String)
    D_TIME = Column(DateTime, default=datetime.now)
    ID = Column(String,nullable=True)
    D_STATUS = Column(String,nullable=True)
    
class Video(Base):
    __tablename__ = "VIDEO"

    VNO = Column(Integer, primary_key=True, autoincrement=True)
    V_TITLE = Column(String(200))
    PATHUPLOAD = Column(String(500))
    FILETYPE = Column(String(50))
    REGDATE = Column(DateTime, default=datetime.now)
    DNO = Column(Integer)  # 왜래키 뺐다.

    

   