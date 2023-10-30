PORT=8529
docker run \
 -d \
 -p 8529:$PORT \
 --env-file .env \
 -v ./data:/var/lib/arangodb3 \
 -v ./dbs:/dbs \
 --name arangodb-instance \
 arangodb/arangodb:3.11.2
echo "ArangoDB started at http://localhost:$PORT"
