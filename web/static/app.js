$(function(){

    $(".loading_icon").hide();
    $(".error_icon").hide();

    var result_window;

    var fileUploadSuccess = function(data){
        console.log("secceed!");

        // var new_tab = window.open("about:blank", '_blank');
        result_window.document.write(data);
        result_window.document.close();

        $(".upload_icon").show();
    };

    var fileUploadFail = function(data){
        $(".loading_icon").hide();
        $(".upload_icon").hide();
        $(".error_icon").show();
        console.log("upload failed!");
    };

    var dragHandler = function(evt){
        evt.preventDefault();
        console.log("drag");
        $("#dropping_mask").css("background", "rgba(0,0,0,0.3)");
    };

    var dragExitHandler = function(evt){
        evt.preventDefault();
        $("#dropping_mask").css("background", "rgba(0,0,0,0)");
    };

    var dropHandler = function(evt){
        evt.preventDefault();
        var files = evt.originalEvent.dataTransfer.files;

        // reset the color
        $("#dropping_mask").css("background", "rgba(0,0,0,0)");

        var formData = new FormData();
        formData.append("ekg_raw_file", files[0]);

        var req = {
            url: "/submit_ekg",
            method: "post",
            processData: false,
            contentType: false,
            data: formData,
            timeout: 20000, // 20 seconds
        };

        result_window = window.open("about:blank", '_blank', "height=" + $(window).height()+200 + ",width=800");
        result_window.blur();
        window.focus();
        var promise = $.ajax(req);
        promise.then(fileUploadSuccess, fileUploadFail);
    };

    var dropHandlerSet = {
        dragover: dragHandler,
        drop: dropHandler,
        dragleave: dragExitHandler
    };

    $("#dropping_zone").on(dropHandlerSet);

    $(document).ajaxStart(function(){
        $(".error_icon").hide();
        $(".upload_icon").hide();
        $(".loading_icon").show();
    });

    $(document).ajaxStop(function(){
        $(".loading_icon").hide();
    });
});
