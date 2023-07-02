# Postgres_tutorial
import json
from typing import List
from fastapi import APIRouter, HTTPException, status, Depends
import pandas as pd
from sqlalchemy import func
from ..preprocessing import process_dataset
from ..dataset_handler import prepare_classifier
from app.config import Settings
from .. import models, schemas, oauth2
from sqlalchemy.orm import Session
from ..database import engine, get_db

# no need with alembic!
models.Base.metadata.create_all(bind=engine)

router = APIRouter(prefix='/home', tags=["HomePage"])

topics = ['Bussiness', 'Education', 'Entertainment', 'News', 'Football']
def estimate_topics():
    df = pd.read_sql_table('posts',
                            engine,
                            columns=[
                                'likes', 'comments', 'shares',
                                'value', 'time', 'timestamp',
                                'topic', 'pagename'
                                ],
                            index_col='index')

    X, _ = process_dataset(df, test_ratio=0.1)
    clf = prepare_classifier('svc', 1234)
    return clf.predict(X)

## When returning a LIST of class instances we can use typing.List variable 
@router.get('/', response_model=List[schemas.PostVote])
def get_posts(db: Session = Depends(get_db),
                 current_user: int = Depends(oauth2.get_current_user)):
    
    results = db.query(models.Post, func.count(models.Vote.post_id).label("votes")).join(models.Vote, models.Vote.post_id == models.Post.index, isouter=True).group_by(models.Post.index).all()
    print(results)
    results = list ( map (lambda x : x._mapping, results) )
    return results


# we must set status to 201
## client-data schema is fed into the function, while api-data schema is fed into the decorator  
@router.post('/', response_model=List[schemas.ReturnedPost])
def retrive_posts(Topic: schemas.Topic, 
                 db: Session = Depends(get_db),
                 current_user: int = Depends(oauth2.get_current_user)):
    with open('app/topics.json','r') as f:
        topics_pages = json.load(f)
    topic = Topic.dict()['topic']
    estimated = Topic.dict()['estimated']
    if topics_pages.keys().__contains__(topic):
        related_topic = models.Post.estimtopic if estimated else models.Post.topic
        related_posts = db.query(models.Post).filter(related_topic == topic).order_by(models.Post.value).all()
        return related_posts
    else:
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE, detail='No such topic')

@router.put('/{id}', response_model=schemas.ReturnedPost)
def update_post(id: int, updated_post: schemas.UpdatePost, db: Session = Depends(get_db), current_user: int = Depends(oauth2.get_current_user)):
    if current_user.id == Settings().admin_id:
        post = db.query(models.Post).filter(models.Post.index == id)
        if post.first() == None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'post with id {id} not found')
        post.update(updated_post.dict())
        db.commit()
        estimtopics = estimate_topics()
        posts = db.query(models.Post).all()
        for id, updpost in enumerate(posts):
            updpost.estimtopic = estimtopics[id]
        db.commit()
        return post.first()
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='access is restricted')

@router.get('/{id}', response_model=schemas.PostVote)
# an integer is required as ID with error log returned to client when not 
# Be ware of mis-matching with other get\post routs (e.g. get('/posts/latest')) 
def get_posts(id: int, db: Session = Depends(get_db),
                 current_user: int = Depends(oauth2.get_current_user)):
    if current_user.id == Settings().admin_id:
        post = db.query(models.Post, func.count(models.Vote.post_id).label("votes")).join(models.Vote, models.Vote.post_id == models.Post.index, isouter=True).group_by(models.Post.index).filter(models.Post.index == id).first()
        return post
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='access is restricted')



@router.delete('/{id}', status_code=status.HTTP_204_NO_CONTENT)
def delet_post(id : int, db: Session = Depends(get_db),
                 current_user: int = Depends(oauth2.get_current_user)):
    if current_user.id == Settings().admin_id:
        post = db.query(models.Post).filter(models.Post.index == id)
        if post.first() == None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='post not found')
        post.delete(synchronize_session=False)
        db.commit() 
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='access is restricted')
