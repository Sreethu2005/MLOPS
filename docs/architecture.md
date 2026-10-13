# Architecture Diagram

```mermaid
flowchart TD
    subgraph External Sources
        UCI[UCI Machine Learning Repository\n(ZIP File)]
    end

    subgraph Data Acquisition
        Ingest[Python/Pandas Ingestion Script]
        LocalStore[Secure Local Landing Area\ndata/raw/input]
        UCI -->|Download| LocalStore
        LocalStore --> Ingest
    end

    subgraph Orchestration
        Airflow[Apache Airflow DAG]
    end

    subgraph Database - PostgreSQL
        direction TB
        subgraph Raw Layer
            RawDB[(raw.raw_hospital_data)]
        end
        
        subgraph Staging Layer
            StagingDB[(staging.staging_hospital_data)]
        end
        
        subgraph Curated Layer
            CuratedP[(curated.patients)]
            CuratedA[(curated.admissions)]
            CuratedD[(curated.diagnoses)]
            CuratedS[(curated.patient_summary)]
            CuratedR[(curated.readmissions)]
        end
        
        subgraph Analytics Layer
            AnalyticViews[[Analytical Views/Data Mart]]
        end
        
        subgraph Audit Layer
            IngestLog[(audit.ingestion_log)]
            ValidLog[(audit.validation_log)]
            RejectLog[(audit.rejected_records)]
        end
    end

    subgraph Dashboard
        Streamlit[Streamlit Dashboard\n(Plotly)]
    end

    %% Flow
    Airflow -->|Triggers| Ingest
    Ingest -->|Load| RawDB
    Ingest -.->|Log| IngestLog
    
    RawDB -->|Validate & Clean| StagingDB
    StagingDB -.->|Log Validation| ValidLog
    StagingDB -.->|Log Rejections| RejectLog
    
    StagingDB -->|Transform & Generate Surrogate Keys| CuratedP
    StagingDB --> CuratedA
    StagingDB --> CuratedD
    StagingDB --> CuratedS
    StagingDB --> CuratedR

    CuratedP --> AnalyticViews
    CuratedA --> AnalyticViews
    CuratedD --> AnalyticViews
    CuratedS --> AnalyticViews
    CuratedR --> AnalyticViews

    AnalyticViews -->|Query| Streamlit
```
