#tar xf redis-stable.tar.gz
#~/redis-stable/src/redis-server &
#tar xf milvus_1.0.tar.gz
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/milvus/lib
#cd milvus/scripts
#sh start_server.sh &
#cd ../../
#wget https://paddlerec.bj.bcebos.com/aistudio/aistudio_paddlerec_rank.tar.gz --no-check-certificate
#tar xf aistudio_paddlerec_rank.tar.gz
#wget https://paddlerec.bj.bcebos.com/aistudio/user_vector.tar.gz --no-check-certificate
#mkdir user_vector_model
#tar xf user_vector.tar.gz -C user_vector_model/
wget https://paddlerec.bj.bcebos.com/aistudio/users.dat --no-check-certificate
wget https://paddlerec.bj.bcebos.com/aistudio/movies.dat --no-check-certificate
wget https://paddlerec.bj.bcebos.com/aistudio/movie_vectors.txt --no-check-certificate
#python3 to_redis.py
#python3 to_milvus.py
