
---
#### Setup MongoDB

* Create an account at mongolab.com

* Sign in and get to the home screen

* Next to "MongoDB Deployments" you'll see three buttons. Click the one that says "Create new".

* For "Cloud provider", select "amazon web services".  For "Location": "Amazon's US East (Virginia) Region (us-east-1)".

* Under "Plan", choose "Single-node" and select "Sandbox" (...it's free).

* For "Database name" write "covertrack".

* Click "Create new MongoDB deployment".

* Back on the home screen, click on "covertrack".

* Click on the "Users" tab and then select "Add a database user".

* Choose a username and password.  Note that in fireworks, the password is stored in plaintext.

* Note that the information shown at the top ("To connect using the shell") contains the database hostname and database port (the number after the colon is the port).

* Run `python initialize.py`
- copy and paste example paths when asked to supply one, except for covertrack which should be in the directory you created in $PI_SCRATCH

* Run `lpad reset`

---
#### Prepare tasks  

FireWorks manages queues to run many jobs in Sherlock.  

```
lpad reset  
python call_fireworks.py YOUR_INPUTARGS_PATH.py  
lpad get_fws  
```
Let's call each tasks as FireTasks.
1. `lpad reset` to remove FireTasks you might have already prepared.  
2. `python call_fireworks` to prepare FireTasks. You can pass inputArgs.py as input arguments. Each FireTasks is associated with ia_path and imgdir. It's like lining balls of fireworks without fire.
3. `lpad get_fws` will check all the FireTasks you have prepared.  
Now you can stack more tasks by passing different input arguments, or launch!

#### launch fireworks

```
tmux
qlaunch -r rapidfire -m 10 --nlaunches infinite
ctrl+b d
```  

FireWorks can submit a number of FireTasks you allow to run at a time. Let's say you have 100 folders; in the example above, you are running 10 jobs at a time.  
It works in an active window, so when you close a terminal it stops working (submitted jobs will be running but FireWorks stops). tmux is just to open a new window where you can detach (ctrl+b, d) and come back (tmux a), so you can work on other things.  
Ctrl+R is a convenient command for history search.
