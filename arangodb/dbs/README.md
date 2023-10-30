Once you copied the dump files here you can restore the database with the following command:
```sh
docker exec -it arangodb-instance cd <db-folder> ./load_db.sh <db-name> tcp://localhost:8529 <db-password>
```
