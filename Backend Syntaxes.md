### For Flask (Basic Flask App):
```
from flask import Flask
app = Flask(__name__)

@app.route('/')
def fucntionName():
	#Argument goes here

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
```

### For PHP:
```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hello PHP</title>
    <link rel="stylesheet" href="styles\style.css">
</head>
<body>



<?php

class Hotdog{
    public $haba;
    //class contructor
    public function __construct($haba)
    {
        $this ->haba = $haba;
    }
    public function print(){
        echo "Masarap ang " . $this->haba . " na hotdog"; 
    }

}

//setting the object variable
$hotdog = new Hotdog("6 inch");

//variable
$name = "Hotdog";
echo "Masarap sumubo ng Hotdog";
echo "<h1>I Love $name !!</h1>" ;

//calling the function
$hotdog->print()

?>
    
</body>
</html>
```