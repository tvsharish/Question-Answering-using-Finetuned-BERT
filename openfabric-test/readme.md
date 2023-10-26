
## Requirement
The current project has the blueprint structure of an AI App. 

Our mission is to implement an 💬NLP chatbot **answering questions about science**. 

Add logic to the `main.py` file inside the `execute` function. 
```python
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:        
        response = '' # <<< --Your magic goes here
        output.append(response)

    return SimpleText(dict(text=output))
```

## Run
The application can be executed in two different ways:
* locally by running the `start.sh` 
* on in a docker container using `Dockerfile` 

## Submission
Your solution must be uploaded on GitHub, and submit us the link in **max 1 week** after receiving the task.

## Note
Keep in mind that this is the project that will be used to evaluate your skills.
So we do expect you to make sure that the app is fully functional and doesn't have any obvious missing pieces.
