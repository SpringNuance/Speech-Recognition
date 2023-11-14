#!/bin/bash

if [ ! -n "$1" ]
then
  echo "Usage: `basename $0` targetdirectory"
  exit 1
fi

if [ -d "$1" ]
then
    echo "Give directory that does not exist yet"
    exit 1
fi


if [ ! -d "/work/courses/T/S/89/5150/submissions/${USER}/prog_assignment.git" ]; then

mkdir -p "/work/courses/T/S/89/5150/submissions/${USER}/prog_assignment.git"
git init --bare /work/courses/T/S/89/5150/submissions/${USER}/prog_assignment.git

fi

git clone /work/courses/T/S/89/5150/general/git/prog_assignment.git $1

cd $1
git checkout -b $USER
git remote add submission /work/courses/T/S/89/5150/submissions/${USER}/prog_assignment.git
git push -u submission $USER

