function WebGLParticles(gl, number) {

    self = this;

    this.gl = gl;
    this.vertexBuffer = this.gl.createBuffer();

    this.numPoints = number || 1000;
    this.vertices = [];
    this.velocities = [];



    for (var i = 0; i < this.numPoints; i++) {
        this.vertices.push(0, 0, 0.85);
        
        var theta=2*Math.random()*Math.PI;
        var phi=Math.random()*Math.PI;
        var r = 1;
        
        var tmp = [r * Math.cos(theta)* Math.sin(phi), 
                   r * Math.sin(theta)* Math.sin(phi),
                   r * Math.cos(phi)];
//        var tmp=[0,0,0];
        this.velocities.push(tmp[0],tmp[1],tmp[2]);
    }
    this.vertices = new Float32Array(this.vertices);
    this.velocities = new Float32Array(this.velocities);



    this.Load = function (modelViewMatrix, perspectiveMatrix, program) {

        self.gl.bindBuffer(self.gl.ARRAY_BUFFER, self.vertexBuffer);

        var vertexPosAttribLocation = self.gl.getAttribLocation(program, "vertexPosition");
        self.gl.enableVertexAttribArray(vertexPosAttribLocation);
        self.gl.vertexAttribPointer(vertexPosAttribLocation, 3, self.gl.FLOAT, false, 0, 0);

        var uModelViewMatrix = self.gl.getUniformLocation(program, "modelViewMatrix");
        self.gl.uniformMatrix4fv(uModelViewMatrix, false, new Float32Array(perspectiveMatrix));

        var uPerspectiveMatrix = self.gl.getUniformLocation(program, "perspectiveMatrix");
        self.gl.uniformMatrix4fv(uPerspectiveMatrix, false, new Float32Array(modelViewMatrix));


    }

//    this.Update = function () {
//
//        for (i = 0; i < self.vertices.length; i++) {
//            self.vertices[i] += 0.001 * self.velocities[i];
//        }
//
//    }


    this.Draw = function () {
		
		console.log(self.vertices[0*3+2]+"   "+self.vertices[(self.numPoints-1)*3+2])

        gl.lineWidth(2.5);
        self.gl.clear(self.gl.COLOR_BUFFER_BIT | self.gl.DEPTH_BUFFER_BIT);
        self.gl.bufferData(self.gl.ARRAY_BUFFER, self.vertices, self.gl.DYNAMIC_DRAW);
        self.gl.drawArrays(self.gl.GL_POINTS, 0, self.numPoints);
        self.gl.flush();

    }


        this.Update = function () {
    
            var screen = document.getElementById("webGLCanvas");
            var ratio = canvas.width/canvas.height;
            
            
    
            var i, n = self.vertices.length,
                p, bp;
            
            for (i = 0; i < self.numPoints; i += 2) {
                bp = i * 3;
                // copy old positions
                self.vertices[bp] = self.vertices[bp + 3];
                self.vertices[bp + 1] = self.vertices[bp + 4];
    
                // inertia
                self.velocities[bp] *= self.velocities[bp + 2];
                self.velocities[bp + 1] *= self.velocities[bp + 2];
    
                // horizontal
                p = self.vertices[bp + 3];
                p += self.velocities[bp];
                if (p < -ratio) {
                    p = -ratio;
                    self.velocities[bp] = Math.abs(self.velocities[bp]);
                } else if (p > ratio) {
                    p = ratio;
                    self.velocities[bp] = -Math.abs(self.velocities[bp]);
                }
                self.vertices[bp + 3] = p;
    
                // vertical
                p = self.vertices[bp + 4];
                p += self.velocities[bp + 1];
                if (p < -1) {
                    p = -1;
                    self.velocities[bp + 1] = Math.abs(self.velocities[bp + 1]);
                } else if (p > 1) {
                    p = 1;
                    self.velocities[bp + 1] = -Math.abs(self.velocities[bp + 1]);
    
                }
                self.vertices[bp + 4] = p;
    
                if (touched) {
                    
                    var dx = touchX - self.vertices[bp],
                        dy = touchY - self.vertices[bp + 1],
                        d = Math.sqrt(dx * dx + dy * dy);
                    if (d < 2) {
                        if (d < .03) {
                            self.vertices[bp + 3] = (Math.random() * 2 - 1) * ratio;
                            self.vertices[bp + 4] = Math.random() * 2 - 1;
                            self.velocities[bp] = 0;
                            self.velocities[bp + 1] = 0;
                        } else {
                            dx /= d;
                            dy /= d;
                            d = (2 - d) / 2;
                            d *= d;
                            self.velocities[bp] += dx * d * .01;
                            self.velocities[bp + 1] += dy * d * .01;
                        }
                    }
                }
            }
    
        }


}