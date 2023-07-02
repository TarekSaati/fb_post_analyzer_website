# The models file contains data we want to create in DB
from sqlalchemy import Integer, Column, String, Text, TIMESTAMP, ForeignKey, BigInteger, DOUBLE_PRECISION, SmallInteger
from sqlalchemy.sql.expression import text as txt
from .database import Base

class Post(Base):
    __tablename__ = 'posts'
    index = Column(BigInteger, primary_key=True, nullable=False)
    text = Column(Text, nullable=False)
    likes = Column(BigInteger, nullable=True)
    comments = Column(BigInteger, nullable=True)
    shares = Column(BigInteger, nullable=True)
    value = Column(SmallInteger, nullable=True)
    time = Column(TIMESTAMP, server_default=txt('now()'))
    timestamp = Column(DOUBLE_PRECISION, nullable=False)
    pagename = Column(Text, nullable=False)
    topic = Column(Text, nullable=True)
    estimtopic = Column(Integer, nullable=True)


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, nullable=False)
    firstName = Column(String, nullable=False)
    lastName = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)

class Vote(Base):
    __tablename__ = 'votes'
    user_id = Column(BigInteger, ForeignKey('users.id', ondelete="CASCADE"), primary_key=True)
    post_id = Column(BigInteger, ForeignKey('posts.index', ondelete="CASCADE"), primary_key=True)
    
    