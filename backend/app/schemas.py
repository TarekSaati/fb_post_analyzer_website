from . import models
from pydantic import BaseModel, EmailStr, conint
from datetime import datetime

# The front-end client must adapt posts to certain schemas depending on the HTTP request
class BasePost(BaseModel):
    index: int
    pagename: str
    text: str
    
class UpdatePost(BaseModel):
    index: int
    topic: str

class BaseUser(BaseModel):
    email: EmailStr  

class UserOut(BaseUser):
    id: int
    email: EmailStr
    class Config():
        orm_mode = True

class ReturnedPost(BasePost):
    time: datetime
    topic: str
    class Config:
        orm_mode = True


class CreateUser(BaseUser):
    firstName: str
    lastName: str  
    password: str   

class Topic(BaseModel):
    topic: str
    estimated: bool

class UpdateUser(BaseUser):
    firstName: str
    lastName: str 
    password: str

class LoginCreds(BaseModel):
    email: EmailStr
    password: str

class OauthCreds(BaseModel):
    username: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    id: int

class Vote(BaseModel):
    post_id: int
    # constrained integer <= 1 e.i. {0, 1}
    dir: conint(le=1)

class VoteData(BaseModel):
    user_id: int
    post_id: int

    class Config:
        orm_mode = True

class PostVote(BaseModel):
    Post: ReturnedPost
    votes: int

    class Config:
        orm_mode = True
