import logging
import azure.functions as func

from dependencies import Dependencies, get_dependencies

app = func.FunctionApp()


@app.blob_trigger(arg_name="blob", path="input", connection="AzureWebJobsStorage") # type: ignore
def BlobTriggerFunction(blob: func.InputStream):
    NeuralNetworkEvaluator(blob, get_dependencies())

def NeuralNetworkEvaluator(blob: func.InputStream, d: Dependencies):
    logging.info("BlobTriggerFunction - Starting...")
    
    logging.info("BlobTriggerFunction - Parsing to network...")
    network = d.blob_network_parser.parse(blob)
    
    logging.info("BlobTriggerFunction - Evaluating network...")
    d.evaluation.evaluate(network)
    
    logging.info("BlobTriggerFunction - Uploading network to blob storage...")
    d.output_blob_client.upload_blob(network)
    
    logging.info("BlobTriggerFunction - Finished.")