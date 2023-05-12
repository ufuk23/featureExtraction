import os
import pymssql
from configparser import ConfigParser

# Environment params is mandatory
configfile = os.environ['configfile']

if configfile == None:
    print("WARNING: CONFIG FILE IS MANDATORY")

print("config file name: %s" % configfile)

#Get the configparser object
config_object = ConfigParser(interpolation=None)

# path
path = "/home/docker/muesconfig/" + configfile

#Read config
config_object.read(path)

#Get the password
dbinfo = config_object["DATABASE"]
print("Database is {}".format(dbinfo["database"]))
print("User is {}".format(dbinfo["user"]))
print("Password is {}".format(dbinfo["password"]))
print("ip is {}".format(dbinfo["ip"]))
print("select_query is {}".format(dbinfo["select_query"]))
print("update_query_success is {}".format(dbinfo["update_query_success"]))
print("update_query_failure is {}".format(dbinfo["update_query_failure"]))

print(type(dbinfo["update_query_failure"]))
print("str {} aaaa".format("ufuk"))
p=dbinfo["update_query_failure"].format("asd")


global conn
conn = pymssql.connect(server=dbinfo["ip"], port=dbinfo["port"], user=dbinfo["user"], password=dbinfo["password"], database=dbinfo["database"])

cursor = conn.cursor()
cursor.execute(dbinfo["select_query"])

print('ok')

