import pyodbc
from tensorflow.keras.preprocessing import image

conn = pyodbc.connect("DRIVER={ODBC Driver 17 for SQL Server};SERVER=172.17.20.41;PORT=1433;UID=muesd;PWD=Mues*dev.1;DATABASE=mues_dev")
cursor = conn.cursor()
cursor.execute("SELECT TOP 100 F.ESER_ID, F.FOTOGRAF_PATH FROM ESER_FOTOGRAF F LEFT JOIN ESER E ON F.ESER_ID = E.ID WHERE ANA_FOTOGRAF = 1 AND E.AKTIF = 1 AND E.SILINMIS = 0 order by F.ESER_ID")

records = cursor.fetchall()
print("Total rows are:  ", len(records))

fs = "/mnt/muesfs/mues-images/image/ak/"

for row in records:
    print("id: ", row[0], row[1])
    print("\n")
    try:
        img = image.load_img(fs + row[1])
        img = image.img_to_array(img)
        print(img)
    except:
        print("file not found")

    # get path from db
    # preprocess the photo
    # give the photo and get the vector from the model
    # TODO: get the featureVector
    # save the vector to the Milvus DB
    conn.execute("UPDATE ESER_FOTOGRAF set DOLASIM_KOPYASI_PATH = 'X' where ANA_FOTOGRAF=1 AND ESER_ID=" + str(row[0]));

conn.commit()
conn.close()
