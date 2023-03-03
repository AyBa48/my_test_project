#!/bin/sh

####update encounter-notes ########
cd /home/ayouba/spiced/nigela-network-encounter-notes/
git pull origin master

echo '######-------#########'
echo pull finished. Now start update student-code file 
echo '######-------########'

#cd /home/ayouba/

##### update student code ##########
#cp "$(ls -t /home/ayouba/spiced/nigela-network-encounter-notes/* | tail -1)" /home/ayouba/spiced/nigela-network-student-code/

#rsync --progress -r -u /home/ayouba/spiced/nigela-network-encounter-notes/* /home/ayouba/spiced/nigela-network-student-code/

#rsync -av --exclude ={'*.txt','.pdf', '.png', '.jpeg', '.md', '.csv', '.xlsx', '.json'} /home/ayouba/spiced/nigela-network-encounter-notes/ /home/ayouba/spiced/nigela-network-student-code/

# every week new folder is added to encounter note - how to copy only code file from subfolder "name of subfolder is not know in advance ???"
rsync -a  --include '*/'--include '*.ipynb' --exclude '*' ~/spiced/nigela-network-encounter-notes/week*/  ~/spiced/nigela-network-student-code/


