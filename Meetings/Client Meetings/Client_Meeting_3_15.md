# Client Meeting Minutes
## Date
**15 March 2024**
## Participant
**Client:** Mario Andre Munoz Acosta

**Mentor:** Ben Golding

**Attendees:**
- Kian Dsouza
- Xin Xiang
- Yusuf Berdan Guzel
- Junheng Chen
- Nathan Harvey
- Dong Hyeog Jang

## Recording
https://drive.google.com/drive/folders/19kMC5elHTws0AJuGZRACMb96zdyZbsBh?usp=sharing

## Agenda
- Discuss on the system to clear any ambiguity
- Identify and prioritise tasks that need to be done

## System Overview:
- Provide statistics on the problem to determine which algorithm fits best to solve the problem.
- Generate a visual representation of the space problem, showing the entropy of the dataset.
- 3 parts:
    - Landing Page
        - Put all the research conducted (repository of data of problems, Library sections).
    - Online Tool (MATLAB)
        - Given metadata sets (table of statistical features and algorithmic performance), it will generate visualization and model that allow user to explore.
    - Management of Users accessing the tools
        - Consists of 2 VM from Melbourne Cloud
            - WEB (2 core): HTML code, has access to the database that contains all experiments, and user management.
            - COMPUTATION. (16 cores): Responsible for processing jobs submitted
        - MySQL for database (need verification).
        - Display color-coded graphs to indicate selected algorithm's goodness/badness on the dataset.
        - Download options
            - Some auto-generated by Highcharts
            - MATHLAB graph and Instance Space are by MATHLAB code
            - Graphs are generated with the Index Colored Scheme/Pallete
- The current system is partially integrated with the UniMelb system.
    - The user management system is not part of the UniMelb system
    - Due to an upgrade in the MelbUni system, some of the features became incompatible.
        - E.g., cannot recover password
- Landing Page and Management are frontend part written in JS
- API b/w frontend and tool:
    - MATLAB creates csv file and frontend will read this file.

## User Registration Process:
- User sends a registration request
- Admin review the request
- Once approved, an e-mail is sent to the user notifying the access grant

## User Dashboard
- Part of User Management Part.
- Submit jobs to a queue, tables showing jobs that are run (Submitted jobs, completed jobs, running jobs, queued, failed, canceled jobs)
- Issue: jobs disappearing

## Instance Analysis Space
- Choose from the library or custom problem
- Select algorithms to use in the analysis, select features to use in the analysis
- Select Optimization and Performance criteria
- Navigate to a page for more detail tuning (more parameter tunings)
- Once submitted, the MATLAB log will be displayed (this will appear in the dashboard even if you leave the page during the process)
- Result Page: Graph showing results (color-coded). Ability to select graph to visualize, select the color scale, select feature. Instance Space 2D coordinate (Projection Matrix x Feature vector). Also shows footprint analysis and average performance analysis.
- Results can be downloaded
- The same data + same setting (parameters, data) will output the same result.

## MATILDA
- Seperated into components (PRELIM, SIFTED (feature selection algorithm), PILOT, CIOISTER, TRACE, PYTHIA).
    - PRELIM: Prepares the meta- data by specifying a binary measure of “good” performance
    - SIFTED: Selects a subset of relevant features by considering correlations with algorithm performance and eliminating redundancies. Can be slow due to optimization problems.
    - PILOT: Bring down high dimension space to 2D space
    - CLOISTER: checks boundaries for the space
    - TRACE: allows us to work with 3d
    - PYTHIA: SVM which generates predictions
    - PRELIM -> SHIFTED -> PILOT 
        -> CIOISTER or 
        -> Trace or 
        -> PYTHIA
- No pre-processing on raw instances.
- Work on the latest branch from GitHub
- Does some checking on the data but not robust.
    - Sometimes causing crashes due to things like NaN values etc...

## Priorities
- MATLAB code migration to python3
    - Main components of MATHILDA 
- Checking why the Slurm job management system crashes.
- Improve Performance/Computation
- Dashboard features
* Focus on the backend part first and we will move on to the frontend part to seek improvements after the backend is done.

## Documentation Requirement:
- ReadME/Markdown is fine
    - Like the one in the InstanceSpace repo.
- Diagrams may be required depending on the complexity of the systems.
    - E.g., Class diagrams

## Slurm
- Parallel job management system from Apache
- When the submit button is pressed, a script will run to submit a job request through Slurm to VMs.
- Need to investigate why it is crashing.

## Future plans:
- Adding new ideas. E.g., Projections
- Increase the availability of tools 
- Users will also be able to download the tool locally and use it
    - Publically funded research, hence will be open source project.

## Side Notes:
- Contact the client if we have questions about the project code
- Client meeting every two weeks (extra meeting during the early phase of the project)
- Extra support for the frontend part from Dr.Neelofar
    - Need to give GitLab username to gain access to the frontend part.
- Communication channel: e-mails
- We are flexible in making changes to API endpoints.
- We need to explain and suggest to the clients any framework we are willing to use.

## Action Items
- Documentation of requirement elicitation and analysis
    - Collaborative action required from team.