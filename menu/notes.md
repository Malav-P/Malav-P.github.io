---
layout: default_wide
title: Notes
permalink: /notes
---

<style>
* {
  box-sizing: border-box;
}


/* html{
    font-size: 100.0%;    
}  */

/* body{
      margin:  0 auto; 
      padding: 1em;
      color: #444; 
      font-family: Tahoma, Verdana,   Segoe, sans-serif;
      font-style: normal; 
      font-size: 1em;              
      max-width: 1200px; 
      background: #FFFFFF;
} */


/* Create three equal columns that floats next to each other */
.column {
  margin:  0 auto; 
  float: left;
  max-width: 30%;
  padding: 10px;  
  margin:  0 auto;   
}

/* Clear floats after the columns */
.row:after {
  content: "";
  display: table;
  clear: both;
}

h2 {text-align:center;}

li:not(:last-child) { 
   margin-bottom: 5px;  
}

/* div.box {border:1px solid #D3D3D3; margin:  10px auto;} */
div.box {margin:  10px auto;}

</style>

<DIV style="margin:0 auto; max-width: 1096px; ">
    <div class="row">
        <div class="column">    
            <DIV class="box">
                <h2>Math</h2>
                    <ul style="list-style-type:none;">
                        {% for note in site.notes %}
                            <li><a href="{{ note.url }}"> {{ note.title }}</a></li>
                        {% endfor %}
                    </ul>
            </DIV>
        </div>
        <div class="column">    
            <DIV class="box">
                <h2>Coding</h2>
                    <ul style="list-style-type:none;">
                        {% for code in site.coding %}
                            <li><a href="{{ code.url }}"> {{ code.title }}</a></li>
                        {% endfor %}
                    </ul>
            </DIV>
        </div>
        <div class="column">    
            <DIV class="box">
                <h2>More Stuff</h2>
            </DIV>
        </div>
    </div>
</DIV>