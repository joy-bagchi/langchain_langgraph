import utils
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# Define a resource with attributes that describe your application
# Here, we're setting the service name to identify what is being traced
resource = Resource(attributes={
    "service.name": "Test Service"
})

# Set up the tracer provider that will manage and provide tracers
# 'TracerProvider' is initialized with the resource we just defined
trace.set_tracer_provider(TracerProvider(resource=resource))

# Create a console exporter to output spans to the console for demonstration purposes
# In a real-world scenario, you might use an OTLP exporter to send spans to a tracing system
console_exporter = ConsoleSpanExporter()

# Set up a span processor to handle the spans
# SimpleSpanProcessor sends each span to the exporter as soon as it is finished
span_processor = SimpleSpanProcessor(console_exporter)

# Add the span processor to the tracer provider to start processing spans immediately
trace.get_tracer_provider().add_span_processor(span_processor)

# Obtain a tracer for the current module to create and manage spans
tracer = trace.get_tracer(__name__)

def retrieve(query, fail=False):
    # Start a span to trace the retrieval process
    with tracer.start_as_current_span("retrieving_documents") as span:
        # Log the event of starting retrieval
        span.add_event("Starting retrieve")
        # Record the input query as an attribute for visibility
        span.set_attribute("input.query", query)
        try:
            # Simulate a retrieval failure if 'fail' is True
            if fail:
                raise ValueError(f"Retrieve failed for query: {query}")

            # Simulated list of retrieved documents
            retrieved_docs = ['retrieved doc1', 'retrieved doc2', 'retrieved doc3']
            # Record details about each retrieved document
            for i, doc in enumerate(retrieved_docs):
                span.set_attribute(f"retrieval.documents.{i}.document.id", i)
                span.set_attribute(f"retrieval.documents.{i}.document.content", doc)
                span.set_attribute(f"retrieval.documents.{i}.document.metadata", f"Metadata for document {i}")
        except Exception as e:
            # If an exception occurs, log and set the span status to indicate an error
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            # Reraise the exception for handling by the caller
            raise

        # Mark the span as successful if no error was raised
        span.set_status(Status(StatusCode.OK))
        return retrieved_docs

# The tracer is configured to show the span in the output
retrieve("Test")

def format_documents(retrieved_docs):
    # Start a span to trace the formatting of documents
    with tracer.start_as_current_span("call_format_documents") as span:
        # Log the event for initiating document formatting
        span.add_event("Calling format_documents")
        # Record the number of documents being formatted
        span.set_attribute("input.documents_count", len(retrieved_docs))

        t = ''
        for i, doc in enumerate(retrieved_docs):
            t += f'Retrieved doc: {doc}\n'
            # Log an event for each processed document
            span.add_event(f"processed document {i}", {"document.content": doc})

        # Mark the span as successful after formatting documents
        span.set_status(Status(StatusCode.OK))
    return t

def augment_prompt(query, formatted_documents):
    # Start a span to trace the prompt augmentation process
    with tracer.start_as_current_span("augment_prompt") as span:
        # Log the event for the beginning of prompt augmentation
        span.add_event("Starting prompt augmentation")
        # Record input details such as the query and document length
        span.set_attribute("input.query", query)
        span.set_attribute("input.formatted_documents_length", len(formatted_documents))

        # Create a prompt that combines the query and formatted documents
        PROMPT = f"Answer the query: {query}.\nRelevant documents:\n{formatted_documents}"

        # Mark the span as successful
        span.set_status(Status(StatusCode.OK))
    return PROMPT

def generate(prompt):
    # Start a span to trace the text generation based on the prompt
    with tracer.start_as_current_span("generate") as span:
        # Log the event for starting text generation
        span.add_event("Starting text generation")
        # Record the prompt being used for generation
        span.set_attribute("input.prompt", prompt)

        # Simulate the text generation process
        generated_text = f"Generated text for prompt {prompt}"

        # Mark the span as successful after text generation
        span.set_status(Status(StatusCode.OK))
    return generated_text

def rag_pipeline(query, fail = False):
    # Start a span to trace the entire RAG pipeline process
    with tracer.start_as_current_span("rag_pipeline") as span:
        try:
            # Step 1: Retrieve documents based on the query
            retrieved_docs = retrieve(query, fail = fail)
            # Step 2: Format the retrieved documents
            formatted_docs = format_documents(retrieved_docs)
            # Step 3: Augment the query with relevant documents to form a prompt
            prompt = augment_prompt(query, formatted_docs)
            # Step 4: Generate a response from the augmented prompt
            generated_response = generate(prompt)

            # Mark the span as successful when all steps are completed
            span.set_status(Status(StatusCode.OK))
            return generated_response
        except Exception as e:
            # If any step raises an exception, set the span status to error
            span.set_status(Status(StatusCode.ERROR, str(e)))
            # Reraise the exception for external handling
            raise

# Trace example 1
response = rag_pipeline("This is a test query", fail = False)
print(response)

# Trace example 2
response = rag_pipeline("This is a test query", fail = True)
print(response)