function WebGLShaderProgram(gl) {

    this._linked = false;
    this._vShaderRead = false;
    this._fShaderRead = false;
    this.gl = gl;
    this._program;


    //
    // PUBLIC
    //

    this.GetVertexShader = function () {
        this.vertexShader = GetShader("shader-vs", this.gl.VERTEX_SHADER);
        this._vShaderRead = true;
    }

    this.GetFragmentShader = function () {
        this.fragmentShader = GetShader("shader-fs", this.gl.FRAGMENT_SHADER);
        this._fShaderRead = true;
    }

    this.Link = function () {


        if (!this._vShaderRead) GetVertexShader();
        if (!this._fShaderRead) GetFragmentShader();


        //    Create a shader program. 
        this._program = this.gl.createProgram();
        this.gl.attachShader(this._program, this.vertexShader);
        this.gl.attachShader(this._program, this.fragmentShader);
        this.gl.linkProgram(this._program);



        if (!this.gl.getProgramParameter(this._program, this.gl.LINK_STATUS)) {
            alert("Unable to initialise program");
            this.gl.deleteProgram(this._program);
            this.gl.deleteShader(this.vertexShader);
            this.gl.deleteShader(this.fragmentShader);
            return;
        }

        this._linked = true;
    }

    this.Use = function () {

        if (!this._linked) this.Link();

        this.gl.useProgram(this._program);

    }

    this.GetProgram = function () {
        return this._program;
    }



    //
    // PRIVATE
    //

    function GetShader(id, type) {

        //    Load the fragment shader that's defined in a separate script
        var shaderScript = document.getElementById(id);
        var shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, shaderScript.text);
        this.gl.compileShader(shader);
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            alert("Couldn't compile the "+( (type==this.gl.VERTEX_SHADER)?"vertex":"fragment" )+" shader");
            this.gl.deleteShader(shader);
            return;
        }

        return shader;

    }


}