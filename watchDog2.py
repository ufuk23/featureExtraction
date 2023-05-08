import sys
import pymssql
from os import listdir
from os.path import isfile, join
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('log-watchdog.log')
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


conn = None

def connect_to_db():
    global conn
    try:
        logger.info('OK')
        print ('Argument List:', str(sys.argv))
        print(sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        conn = pymssql.connect(server=str(sys.argv[1]), port=str(sys.argv[2]), user=str(sys.argv[3]), password=str(sys.argv[4]), database=str(sys.argv[5]))
        #conn = pymssql.connect(server='10.1.37.177', port='1033', user='muest', password='Mues*test.1', database='mues_test')
        print('DB connected successfully')
        logger.info('DB OK')
    except Exception as e:
        print(e)
        logger.error(e)

#function to return files in a directory
def fileInDirectory(my_dir: str):
    onlyfiles = [f for f in listdir(my_dir) if isfile(join(my_dir, f))]
    return(onlyfiles)


#function comparing two lists
def listComparison(originalList: list, newList: list):
    differencesList = [x for x in newList if x not in originalList] #Note if files get deleted, this will not highlight them
    print(*differencesList)
    return(differencesList)


def doThingsWithNewFiles(newFiles: list):
    print(f'I would do things with file(s) {newFiles}')
    logger.info(newFiles)



def fileWatcher(watchDirectory: str, pollTime: int):

    connect_to_db()

    while True:
        if 'watching' not in locals(): #Check if this is the first time the function has run
            previousFileList = fileInDirectory(watchDirectory)
            watching = 1

        time.sleep(pollTime)

        newFileList = fileInDirectory(watchDirectory)

        fileDiff = listComparison(previousFileList, newFileList)

        previousFileList = newFileList
        if len(fileDiff) == 0: continue
        doThingsWithNewFiles(fileDiff)

fileWatcher("/mnt/muesfs/mues/temp/", 5)
