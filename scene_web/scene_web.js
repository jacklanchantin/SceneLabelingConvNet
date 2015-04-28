if (Meteor.isClient) {
    Template.submit.events({
        'click button': function () {
            var link = $("#in").val();
            Meteor.call("getPicture", link, function(error, res) { 
                if (error) alert("Something went wrong!");
                var elem = "<img width=600px src=\""+res+"\" / >";
                //$("#image").prepend(elem);
                Session.set("Image", elem);
            });
        }
    });
    Template.submit.rendered = function() {
        var elem = Session.get("Image");
        if (elem) $("#image").prepend(elem);
    }
}

if (Meteor.isServer) {
    Meteor.startup(function () {
        id = 0;
    });
    Meteor.methods({
        getPicture: function(link) { 
            var exec = Npm.require('child_process').exec;
            id = id+1; 
            var cmd = 'cd $HOME/Documents/Scene-Labeling-Conv-Net/scene_web; sh run_model.sh '+link+' '+id;
            console.log(cmd);
            child = exec(cmd, function(error, stdout, stderr) {
                console.log('stdout: ' + stdout);
                console.log('stderr: ' + stderr);

                if(error !== null) {
                    console.log('exec error: ' + error);
                }
            });
            return "/"+id+".jpg"; 
        }
    })
}
