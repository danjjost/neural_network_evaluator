import azure.functions as func

from dependencies import Dependencies, get_dependencies

app = func.FunctionApp()


@app.blob_trigger(arg_name="myblob", path="input", connection="AzureWebJobsStorage") # type: ignore
def BlobTriggerFunction(blob: func.InputStream, d: Dependencies = get_dependencies()):
    NeuralNetworkEvaluator(blob, d)

def NeuralNetworkEvaluator(blob: func.InputStream, d: Dependencies):
    network = d.blob_network_parser.parse(blob)
    
    d.evaluation.evaluate(network)
    
    d.output_blob_client.upload_blob(network)