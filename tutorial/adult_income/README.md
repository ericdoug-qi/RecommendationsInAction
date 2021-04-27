

saved_model_cli run --dir 1616312445 --tag_set serve --signature_def="predict" --input_examples='examples=[{"age":[46.], "education_num":[10.], "capital_gain":[7688.], "capital_loss":[0.],"hours_per_week":[38.]}, {"age":[24.], "education_num":[13.], "capital_gain":[0.], "capital_loss":[0.], "hours_per_week":[50.]}]'



```
docker pull tensorflow/serving
```

docker run -p 8500:8500 -p 8501:8501 --name adult_export_model --mount type=bind,source=/Users/ericdoug/Documents/mydev/RecommendationsInAction/recommender/recommender/framework/tf2/models/adult_export_model,target=/models/adult_export_model -e MODEL_NAME=adult_export_model -t tensorflow/serving

pip install tensorflow-serving-api