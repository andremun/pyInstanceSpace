# Client Meeting Minutes
## Date
**27 March 2024**
## Participant
**Client:** 
- Mario Andre Munoz Acosta 
- Jeffrey Christiansen    
    -   Email: jeffreyc1@unimelb.edu.au 
    -   Github name:jeffrey-chr
- Vivek Katial            
    -   (https://github.com/vivekkatial) 
    -   Github name:vivekkatial
**Mentor:** Ben Golding

**Attendees:**
- Kian Dsouza
- Xin Xiang
- Yusuf Berdan Guzel
- Junheng Chen
- Nathan Harvey
- Cheng Ze Lam
- Jiaying Yi

## Recording
https://unimelb.zoom.us/rec/share/LEZPM-RmQ9Izj4d97GHxigIpgE99dcbwOm_Q4ZcKnTPwnZLI-vQQEPZNFkeTycc.xlIUEJE84C2HNd0y
Passcode: L?83W$z? 

## Agenda 
- Clarification regarding backend and frontend  
- Clarification regarding the user types
- Walkthough the communication MATLAB function called
## Agenda Items
### Clarification communication regarding backend and frontend  
- The web collect information, then store in folder,then details in
Json file. The data uploaded created Slurm scirpt
    - Slurm is framework for submitting job on HPC cluster(bash script like)
- The VM has N num cores, assign 1-2 cores that initiate instance of matlab machine
- csv files generated after done, loaded by the java interface presented as graphs
- move away from visualization -> dashboard, ideally R shiny like
- better integration and presentation 
- slightly different approach from current but simply the process
- **Priority** Modular->python, then replicate the result
- not bound lagency code
### Deadline for transfer backend
- PO: 6-7 wks
### user interface 
- least defined part of the specification
- try to sprint early enough
### open to Astro?
- as long as its able to extend
### Clarification regarding the user types 
- cuurently 100-200 users academic 
- request from company, paid to uni for analysis 
- code is open-source, the replicate the result
- academic, not commercialize 
#### Type of user
- 3 admins
- Superuser -> accept the acct
- wait Neelofar(further discuss) 
- MRC(melb research cloud)
- the application should be able to spin up before user interface
- implment: one admin is able to spin up
- internel Metida: cc the ticket[require Kate]
- try to do everything locally working then VM[docker]
- portable EVM, doesn't even have to be on MRC
### Who do we need to contact for uni authentication
- not clear
### whats in the gitlab.web
- website, housekeeping jobs to the machine
### Conduct any testing for existing backhand libraries?
- none for system test
- replicate the result from the offical web
- set up the pytest for each modular
- use existing output&input as a test case
### Walkthough the communication MATLAB function called
- slurm[define computing resource]
- matlab(buildIS.m path) -> prelim ->sifted -> pilot -> cloisten -> pythid -> trace
- out from each component
- then one script generate pngs, the another script generate csv
### Out product is like a cloud solution, python interact with slurm?
- micro service related to each of these requirements
- call the pilot API/prelim API.
- **desired**: option for stepping out 
### What is the OUT folder
- info from single component
- preserve the tree construct is desirable
### exploreIS.m
- before called test
- use prejection to calc
- prelim [preprocess feature] based on existing model,
then called Pilot to get the projection -> Json [not priority]
### set up Github?
- **client want access**
- wanna use fastAPI
### Astro
- research team lack of js skillset
- require less coding 
### Refactor Java backend
- as long as maintaining the functionality is fine 
- require instruction readme
- looking into docker
### Machine learning  
- pythid [choose the ml model]projection
- feasure alg[RF]
- user select the model
- metadata from web->instance lib questin->cls
- require the label
- SIFTED[least efficient]
### current api for the graph generated 
- highchart 
- use newest verson py
### roadmap of project
- repo setup,stub python modular finished during ester break
- sprint2 -> implementation
- backend(1st) connect java frontend(2rd) 
- ideally skeatching out replacement of the frontend look like, less back and forward
### Login function
- nice to have
- utilze the PR for traceability
### Level of doc
- Current is pretty standard
- Don't document math section
- instruction for run
- component interaction
### Action Items
- ask permission adding two client in
- arrange meeting with Neelofar
