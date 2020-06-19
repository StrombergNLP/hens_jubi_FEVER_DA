package com.function;

import com.microsoft.azure.functions.ExecutionContext;
import com.microsoft.azure.functions.HttpMethod;
import com.microsoft.azure.functions.HttpRequestMessage;
import com.microsoft.azure.functions.HttpResponseMessage;
import com.microsoft.azure.functions.HttpStatus;
import com.microsoft.azure.functions.annotation.AuthorizationLevel;
import com.microsoft.azure.functions.annotation.FunctionName;
import com.microsoft.azure.functions.annotation.HttpTrigger;

import com.function.GoldenDocumentRetriever;

import java.util.Optional;

/**
 * Azure Functions with HTTP Trigger.
 */
public class Function {
    /**
     * This function listens at endpoint "/api/HttpExample". Two ways to invoke it using "curl" command in bash:
     * 1. curl -d "HTTP Body" {your host}/api/HttpExample
     * 2. curl "{your host}/api/HttpExample?name=HTTP%20Query"
     */
    @FunctionName("evidence-retrieval")
    public HttpResponseMessage run(
            @HttpTrigger(
                name = "req",
                methods = {HttpMethod.GET, HttpMethod.POST},
                authLevel = AuthorizationLevel.ANONYMOUS)
                HttpRequestMessage<Optional<String>> request,
            final ExecutionContext context) {
        context.getLogger().info("Java HTTP trigger processed a request.");

        // Parse query parameter
        final String query = request.getQueryParameters().get("claim");
        final String claim = request.getBody().orElse(query);

        if (claim == null) {
            return request.createResponseBuilder(HttpStatus.BAD_REQUEST).body("Please pass a claim on the query string or in the request body").build();
        } else {
            try { 
                String evidence = GoldenDocumentRetriever.retrieve(claim, context);
                return request.createResponseBuilder(HttpStatus.OK).body(evidence).build();
            } catch (Exception e) {
                context.getLogger().warning(e.toString());
                return request.createResponseBuilder(HttpStatus.BAD_REQUEST).body("Exception in Evidence Retriever: " + e.toString()).build();
            }
        }
    }
}
