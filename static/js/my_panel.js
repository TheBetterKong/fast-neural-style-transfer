/***
 * pannel 相关代码
 */

// pFunc1 根据下拉框展示图片
function changeImage() {
    var styleFileImage = 'img/style/' + document.getElementById('style').value + '.jpg';
    console.log(document.getElementById('style').value);
    console.log(styleFileImage);
    document.getElementById('imageShow').src = "../static/" + styleFileImage;
}

// pFunc2 展示上传的 content 图片
function changeContentImage() {
    function getObjectURL(file) {
        var url = null;
        if (window.createObjcectURL != undefined) {
            url = window.createOjcectURL(file);
        } else if (window.URL != undefined) {
            url = window.URL.createObjectURL(file);
        } else if (window.webkitURL != undefined) {
            url = window.webkitURL.createObjectURL(file);
        }
        return url;
    }
    var f = document.getElementById("up").files;
    var objURL = getObjectURL(f[0]);//这里的objURL就是input file的真实路径
    document.getElementById('contentimageShow').src = objURL;
}

function showStatus() {
    // document.getElementById('status').src = '../static/img/loading1.gif'
    var status = document.createElement('img');
    status.setAttribute('src','../static/img/loading1.gif')
    document.getElementById('status').appendChild(status)
}

$(".clickUpload").on("change","input[type='file']",function(){
	var filePath=$(this).val();
	if(filePath.indexOf("jpg")!=-1 || filePath.indexOf("png")!=-1){
		$(".fileerrorTip").html("").hide();
		var arr=filePath.split('\\');
		var fileName=arr[arr.length-1];
		$(".showFileName").html(fileName);
	}else{
		$(".showFileName").html("");
		$(".fileerrorTip").html("您未上传文件，或者您上传文件类型有误！").show();
		return false
	}
})

$(".picurlbtn").on("change","input[type='file']",function(){
    var filePath=$(this).val();
    if(filePath.indexOf("jpg")!=-1 || filePath.indexOf("jpeg")!=-1 || filePath.indexOf("png")!=-1){
        $("#picture_name").attr("value",filePath);
    }else{
		$("#picture_name").attr("value","仅支持jpg,jpeg,png格式！");
        return false
    }
})