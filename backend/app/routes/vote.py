from typing import List
from fastapi import APIRouter, HTTPException, status, Depends
from .. import models, schemas, oauth2
from sqlalchemy.orm import Session
from ..database import engine, get_db
from ..config import Settings

models.Base.metadata.create_all(bind=engine)

router = APIRouter(prefix='/votes', tags=["Votes"])

@router.post('/', status_code=status.HTTP_201_CREATED)
def vote(vote: schemas.Vote, db: Session = Depends(get_db), 
         current_user: int = Depends(oauth2.get_current_user)):
    
    post = db.query(models.Post).filter(models.Post.index == vote.post_id).first()
    if not post:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'post {vote.post_id} not found')
    existing_vote = db.query(models.Vote).filter(models.Vote.post_id == vote.post_id, 
                                                 models.Vote.user_id == current_user.id)
    if vote.dir == 1:
        if existing_vote.first():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='already voted')
        vote = models.Vote(post_id= vote.post_id, user_id= current_user.id)
        db.add(vote)
    else:
        if not existing_vote.first():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='no vote')
        existing_vote.delete(synchronize_session=False)

    db.commit()
    return {"message": "successfully voted"}

@router.get('/', response_model=List[schemas.VoteData])
def get_votes(db: Session = Depends(get_db),
              current_user: int = Depends(oauth2.get_current_user)):
    if current_user.id == Settings().admin_id:
        votes = db.query(models.Vote).all()
        return votes
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='access restricted')
